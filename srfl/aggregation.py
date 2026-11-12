from dataclasses import dataclass
import torch
import time
from .gua import project_simplex


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys() or any(parameters[n].shape != state[n].shape for n in parameters):
        raise ValueError('Trainable parameter state does not match model')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter))


def flatten(state):
    return torch.cat([p.reshape(-1) for p in state.values()])


def unflatten(vector, template):
    result, offset = {}, 0
    for name, tensor in template.items():
        size = tensor.numel()
        result[name] = vector[offset:offset+size].reshape(tensor.shape).to(tensor.dtype)
        offset += size
    if offset != vector.numel():
        raise ValueError('Vector and template size mismatch')
    return result


def aggregate(global_state, client_states, counts, config):
    if not client_states or len(counts) != len(client_states) or any(n <= 0 for n in counts):
        raise ValueError('Non-empty client states and positive sample counts are required')
    if any(s.keys() != global_state.keys() or any(s[n].shape != global_state[n].shape for n in s) for s in client_states):
        raise ValueError('Client parameter names or shapes differ')
    global_vector = flatten(global_state).double()
    clients = torch.stack([flatten(state).double() for state in client_states])
    if not torch.isfinite(clients).all() or not torch.isfinite(global_vector).all():
        raise ValueError('Non-finite client/global parameters')
    if config['method'] in ('fedavg', 'centralized'):
        if config['method'] == 'centralized' and len(clients) != 1:
            raise ValueError('Centralized mode requires one pooled client')
        weights = torch.tensor(counts, dtype=torch.float64)
        weights /= weights.sum()
        vector = weights @ clients
        return unflatten(vector, global_state), {'method':config['method'], 'weights':weights.tolist()}
    if config['method'] != 'gua':
        raise ValueError('Unknown aggregation method')
    started = time.perf_counter()
    difference = global_vector[None] - clients
    deviation = difference.square().sum(1)
    gradients = 2 * difference / deviation.clamp_min(config['eps'])[:,None]
    gradient_seconds = time.perf_counter()-started
    started=time.perf_counter()
    gram = gradients @ gradients.T
    gram = (gram + gram.T) * 0.5
    gram_seconds = time.perf_counter()-started
    started=time.perf_counter()
    largest = torch.linalg.eigvalsh(gram).max().clamp_min(config['eps'])
    step = 1 / largest
    weights = torch.ones(len(clients),dtype=torch.float64) / len(clients)
    for iteration in range(config['iterations']):
        gw = gram @ weights
        gap = (weights @ gw - gw.min()).clamp_min(0.)
        if gap <= config['tolerance']:
            break
        weights = project_simplex(weights - step * gw)
    solver_seconds = time.perf_counter()-started
    raw_direction = weights @ gradients
    norm = torch.linalg.vector_norm(raw_direction)
    direction = raw_direction
    if config['normalize_direction']:
        direction = direction / norm if norm > config['eps'] else torch.zeros_like(direction)
    eta = config['global_step']
    directional_dot = gradients @ direction
    direction_squared = direction.square().sum()
    safe_limit = 0.
    if direction_squared > config['eps'] and (directional_dot > 0).all() and (deviation > config['eps']).all():
        safe_limit = float((deviation * directional_dot / direction_squared).min())
    if config['safeguard']:
        eta = min(eta, config['safety_fraction'] * safe_limit)
    updated = global_vector - eta * direction
    new_deviation = (updated[None] - clients).square().sum(1)
    objective = weights @ gram @ weights
    diagnostics = {
        'gradient_seconds':gradient_seconds, 'gram_seconds':gram_seconds, 'solver_seconds':solver_seconds,
        'aggregation_device':str(gram.device), 'participating_clients':len(clients), 'parameter_count':global_vector.numel(),
        'gram_bytes':gram.numel()*gram.element_size(), 'gradient_bytes':gradients.numel()*gradients.element_size(),
        'method':'gua', 'weights':weights.tolist(), 'objective':float(0.5*objective),
        'optimality_gap':float((objective-(gram @ weights).min()).clamp_min(0.)),
        'iterations':iteration+1, 'raw_direction_norm':float(norm),
        'simplex_residual':abs(float(weights.sum())-1.),
        'minimum_deviation':float(deviation.min()), 'requested_step':config['global_step'],
        'applied_step':eta, 'safe_step_limit':safe_limit,
        'relative_deviation_change':((deviation-new_deviation)/deviation.clamp_min(config['eps'])).tolist(),
        'trainable_upload_bytes':sum(t.numel()*t.element_size() for s in client_states for t in s.values()),
    }
    return unflatten(updated,global_state), diagnostics
