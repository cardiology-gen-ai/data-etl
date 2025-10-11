Configuration Manager
=====================
Configuration models for the File → Markdown → Index ETL pipeline.

This module defines :pydantic:`Pydantic BaseModel <base_model>` used to configure:

- image extraction and placement (:class:`~src.config.manager.ImageManagerConfig`),
- on‑disk storage folders for inputs/outputs (:class:`~src.config.manager.FileStorageConfig`),
- text splitting pipeline (:class:`~src.config.manager.ChunkingManagerConfig`),
- high‑level preprocessing (:class:`~src.config.manager.PreprocessingConfig`),
- global ETL stack (:class:`~src.config.manager.ETLConfig`) and its loader (:class:`~src.config.manager.ETLConfigManager`).

.. automodule:: src.config.manager
   :members:
   :exclude-members:
   :member-order: bysource
