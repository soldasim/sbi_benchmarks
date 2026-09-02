#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -p cpu

export PATH=/Users/soldasim/.juliaup/bin:/Users/soldasim/.local/bin:/Users/soldasim/miniconda3/condabin:/Users/soldasim/.pyenv/shims:/Users/soldasim/.juliaup/bin:/Library/TeX/Root/bin/universal-darwin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/Library/TeX/texbin:/Applications/Wireshark.app/Contents/MacOS
cd ~/repos/bosip_benchmarks
julia --project=src analyze_bosip.jl
