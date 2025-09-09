package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SimpleChaincode implements a simple chaincode to manage an asset
type SimpleChaincode struct {
	contractapi.Contract
}

// InitLedger adds a base set of assets to the ledger
func (s *SimpleChaincode) InitLedger(ctx contractapi.TransactionContextInterface) error {
	return nil
}

// Set stores a key-value pair in the world state
func (s *SimpleChaincode) Set(ctx contractapi.TransactionContextInterface, key string, value string) error {
	if len(key) == 0 {
		return fmt.Errorf("key must not be empty")
	}
	if len(value) == 0 {
		return fmt.Errorf("value must not be empty")
	}

	return ctx.GetStub().PutState(key, []byte(value))
}

// Get returns the value of the specified key from the world state
func (s *SimpleChaincode) Get(ctx contractapi.TransactionContextInterface, key string) (string, error) {
	if len(key) == 0 {
		return "", fmt.Errorf("key must not be empty")
	}

	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return "", fmt.Errorf("failed to read from world state: %v", err)
	}
	if value == nil {
		return "", fmt.Errorf("the asset %s does not exist", key)
	}

	return string(value), nil
}

// GetAllAssets returns all assets found in world state
func (s *SimpleChaincode) GetAllAssets(ctx contractapi.TransactionContextInterface) ([]string, error) {
	// range query with empty string for startKey and endKey does an open-ended query of all assets in the chaincode namespace.
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var assets []string
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var asset string
		err = json.Unmarshal(queryResponse.Value, &asset)
		if err != nil {
			asset = string(queryResponse.Value)
		}
		assets = append(assets, fmt.Sprintf("Key: %s, Value: %s", queryResponse.Key, asset))
	}

	return assets, nil
}

func main() {
	assetChaincode, err := contractapi.NewChaincode(&SimpleChaincode{})
	if err != nil {
		log.Panicf("Error creating simple chaincode: %v", err)
	}

	if err := assetChaincode.Start(); err != nil {
		log.Panicf("Error starting simple chaincode: %v", err)
	}
}

