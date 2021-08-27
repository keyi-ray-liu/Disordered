#! /bin/zsh

curdir=`pwd`
aup=50
bup=50
seed=0
for a in {1..${aup}}
do
for b in {1..${bup}}
do
L=12
N=1
cases=100
lo=0
hi=11
tun=1
cou=1
x=`echo "0.01 * $a" | bc `
y=`echo "0.01 * $b" | bc`
readdis=0
seed=`echo "$seed + 1" | bc`
decay=0.2
t=1.0
ee=1.0
ne=1.0 #extra - sign is to be expected later in code
z=1
zeta=0.5
ex=0.2
selfnuc=0

echo "$L $N $cases $lo $hi" > ref/inp
echo "$tun $cou $x $y $readdis $seed $decay" > ref/para_dis
echo "$t $ee -$ne $z $zeta $ex $selfnuc" > ref/hamiltonian

dir=${L}L${N}N${ee}ee-${ne}ne${ex}ex
subdir=${tun}tun${cou}cou${x}x${y}y${seed}seed

cdir=cases/${dir}/
if [ ! -d $cdir ]
then
mkdir $cdir
fi

inddir=${cdir}$subdir/
if [ ! -d $inddir ]
then
cp -r ref/ $inddir
cd $inddir
python 1Ddiag.py
cd $curdir 
else
echo "results exist"
fi
done
done
