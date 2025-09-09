#!/bin/sh

function genesisBlock() {
  touch ./DAG/genesisBlock.pkl
  ipfs add ./DAG/genesisBlock.pkl | tr ' ' '\n' | grep Qm
}

# Parse commandline args

## Parse mode

if [[ $# -lt 1 ]] ; then
  printHelp
  exit 0
else
  MODE=$1
  shift
fi

if [ "${MODE}" == "genesis" ]; then
  genesisBlock
else
  printHelp
  exit 1
fi