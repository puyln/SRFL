# References and attribution

SRFL combines SAM adaptation with USGR feature regularization and GUA parameter aggregation. The references below cover segmentation foundation models and federated learning.

- Kirillov et al. *Segment Anything*. ICCV, 2023.
- Liao et al. *Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data*. FedU², 2024.
- Liu, Luo, and Zhu. *FedFMS: Exploring Federated Foundation Models for Medical Image Segmentation*. MICCAI, 2024.

The adapter integration uses the installed SAM image encoder rather than redistributing its source. External code, dependencies, and weights retain their respective licenses.

- [SAM repository](https://github.com/facebookresearch/segment-anything)
- [SAM image encoder](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py)
