export MAP_NAME=5m_vs_6m_lite
export RENDER_GAME=False

python main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m

python main.py --config=iql --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=iql --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=iql --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=iql --env-config=sc2 with env_args.map_name=5m_vs_6m
python main.py --config=iql --env-config=sc2 with env_args.map_name=5m_vs_6m
