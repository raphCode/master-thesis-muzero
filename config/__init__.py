from . import schema

# Global configuration container
# It is empty at first and later filled in-place by populate_config()
# So every other module can conveniently import it directly
C = schema.BaseConfig.placeholder()
