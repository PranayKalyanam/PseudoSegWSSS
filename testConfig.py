from configs.config import get_config

args = get_config()

print("Configuration:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    
    