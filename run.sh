export CUDA_VISIBLE_DEVICES=1 && nohup python highway.py PPO 0 > trainlogs/ppo_0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2 && nohup python highway.py ShieldPPO 0 > trainlogs/shield_ppo_0.log 2>&1 &
