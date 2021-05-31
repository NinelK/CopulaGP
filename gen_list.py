with open("list.sh","w") as f:
  for d in range(1,2):
    name = f'Day{d}'
    exp = f"ST260_{name}_Dataset"
    start = 0 #32
    for i in range(start,20):
      f.write(f"python train.py -start {i} -layers {i+1} -exp {exp}_unconditional\n")
      f.write(f"python train.py -start {i} -layers {i+1} -exp {exp}\n")

