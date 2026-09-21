nohup python ./smoke_multicycle.py --out /mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_temp/smoke_runs/overnight --cycles 5 \
    --members 2 --years 2 --seq 120 --rows 600 --epochs 30 --patience 10 \
    --parallel 1 --score-every 0 --score-first-last \
    --converge-on surrogate --rel-tol 0 > ./overnight_resume.log 2>&1 &
