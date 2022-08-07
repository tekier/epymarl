export MAP_NAME=5m_vs_6m_lite
export RENDER_GAME=False

python main.py --config=iql --env-config=griddly
python main.py --config=qmix --env-config=griddly

#python main.py --config=qmix --env-config=griddly
#python main.py --config=iql --env-config=griddly
#
#python main.py --config=qmix --env-config=griddly
#python main.py --config=iql --env-config=griddly
#
#python main.py --config=qmix --env-config=griddly
#python main.py --config=iql --env-config=griddly
#
#python main.py --config=qmix --env-config=griddly
#python main.py --config=iql --env-config=griddly
