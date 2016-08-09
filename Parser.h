#include <stdio.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <queue>
#include <regex>
#include <iterator>

using namespace std;

struct Pattern {
  vector<double> input;
  vector<double> output;
};

class Dataset {
private:
	bool isSettedArchitecture;
  	int input;
  	int hidden;
  	int output;
	Pattern parseMonk(sregex_iterator ptrTokenIter);
	Pattern parseCsv(sregex_iterator ptrTokenIter);  
protected:
	vector<Pattern> patternVector;
public:
	Dataset();
	~Dataset();
  	void popolate(const char* filename, int input, int hidden, int output);
  	void popolate(const char* filename);
  	void setArchitecture(int input, int hidden, int output);
  	void setHiddenUnit(int n);
	vector<Dataset> split(int kfold);
	void unionDatasets(vector<Dataset> v);
	void print();
	int getInputLayerSize();
  	int getHiddenLayerSize();
  	int getOutputLayerSize();
  	int getSize();
  	Pattern getPattern(int position);
};

inline Dataset::Dataset(){ isSettedArchitecture = false; }

Dataset::~Dataset(){
	patternVector.clear();
}

inline int Dataset::getInputLayerSize() { return this->input; }
inline int Dataset::getHiddenLayerSize() { return this->hidden; }
inline int Dataset::getOutputLayerSize() { return this->output; }
inline int Dataset::getSize() { return this->patternVector.size(); }
inline Pattern Dataset::getPattern(int position) { return patternVector.at(position); }
inline void Dataset::setHiddenUnit(int n) { this->hidden = n; }

void Dataset::setArchitecture(int input, int hidden, int output) {
	this -> input = input;
	this -> hidden = hidden;
	this -> output = output;
	isSettedArchitecture = true;
}

void Dataset::popolate(const char* filename) { 
	if(isSettedArchitecture)
		popolate(filename, input, hidden, output); 
	else 
		cerr << "Architecture not setted\n";
}

void Dataset::popolate(const char* filename, int input, int hidden, int output){
	isSettedArchitecture = true;
	ifstream file(filename);
	if(!file) cerr << "Error: failed to open file " << filename << endl;
	// regex extension("[a-zA-Z0-9\\s]+)(\\.(a-zA-Z0-9)+)");
	//regex numbers("((?:^|\\s)([+-]?[[:digit:]]+(?:\\.[[:digit:]]+)?)(?e[+-]?[[:digit:])(?=$|\\s))");//(R"((?:^|\s|\#|^data_0-9)?([+-]?[[:digit:]]+(?:\.[[:digit:]]+)?)(?=$|\s)?)");
	regex numbers("-?[\\d.]+(?:e-?\\d+)?");
	string buffer;
  	this->input = input;
  	this->hidden = hidden;
  	this->output = output; 
	buffer.erase();
	while(file.good()){
		getline(file, buffer, '\n');
		auto ptrTokenIter = sregex_iterator(buffer.begin(), buffer.end(), numbers);
		auto ptrIterEnd = sregex_iterator();
		if(std::distance(ptrTokenIter, ptrIterEnd)==0) continue;
    	patternVector.push_back(parseCsv(ptrTokenIter));
  	}	    
}

Pattern Dataset::parseCsv(sregex_iterator ptrTokenIter){
	Pattern pattern;
	auto tokenIterEnd = sregex_iterator();
  	int cnt = 0;
  	auto i = ptrTokenIter; 
  	while(i!=tokenIterEnd) {
    	if(cnt<input) pattern.input.push_back(stof((*i).str()));
  		else pattern.output.push_back(stof((*i).str()));
  		i++; cnt++; 
  	}
	return pattern;
}

vector<Dataset> Dataset::split(int kfold){
	long n = this->patternVector.size() / kfold;
    vector<Dataset> v;
    for(int i=0; i<kfold; i++){
    	Dataset d;
    	vector<Pattern> p(make_move_iterator(this->patternVector.begin()),make_move_iterator(this->patternVector.begin()+n));
		d.patternVector = p;
		this->patternVector.erase(this->patternVector.begin(),this->patternVector.begin()+n);
		d.input = this -> input; d.hidden = this -> hidden;
		d.output = this -> output; v.push_back(d);
    } return v;
}

void Dataset::unionDatasets(vector<Dataset> v){
	this->input = v.at(0).getInputLayerSize(); this->hidden = v.at(0).getHiddenLayerSize();
	this->output = v.at(0).getOutputLayerSize(); int position = 0; 
    for(int i=0; i<1; i++){
    	this->patternVector.insert(this->patternVector.end(), v.at(i).patternVector.begin(), v.at(i).patternVector.end());
    }
}

void Dataset::print(){
	for(unsigned i = 0; i < patternVector.size(); i++){
		cout << " input: ";
		for (unsigned j=0; j < patternVector[i].input.size(); j++)
			cout << patternVector[i].input[j] << " ";
		cout << " output: ";
		for (unsigned j=0; j < patternVector[i].output.size(); j++)
			cout << patternVector[i].output[j] << " ";
		cout << endl;
	}
}
