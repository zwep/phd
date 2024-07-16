"""
This code should be copied in the 'direct' package under  direct/data/datasets_config.py
"""

@dataclass
class MyDatasetConfig(BaseConfig):
    ...
    name: str = "MyNew"
    lists: List[str] = field(default_factory=lambda: [])
    transforms: BaseConfig = TransformsConfig()
    text_description: Optional[str] = None