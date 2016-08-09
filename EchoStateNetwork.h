#include "Parser.h"
#include "gnuplot_i/gnuplot_i.hpp"
#include "Eigen/Core"
#include "Eigen/Sparse" //add
#include "Eigen/Eigenvalues"
#include "Eigen/LU"
#include <random>

// #define PRINTS

using namespace std;
using namespace Eigen;

class ESN {
public:
    ESN();
    ESN(Dataset train, Dataset test, double sparsity, int washout, double spectralRadius, double leakingRate, double reg);
    void createNetwork();
    void createInput();
    void createReservoir();
    void compute();
    void predict(Dataset blind, const char* output);
    double getError();
    void saveTestPlot(std::string filename);
    void setReservoirSize(int n);
    void saveWeights(const string path);
    void restoreWeights(const string path);
private:
    double MSEtrain;
    double MSEtest;
    double spectralRadius;
    double leakingRate;
    double reg;
    double sparsity;
    int washout;
    vector<double> plotTestVecX,plotTestVecY,plotTestVecT;
    Dataset trainDataset;
    Dataset testDataset;
    VectorXd inputLayer; // u(n)
    VectorXd internalState; // x(n)
    VectorXd outputLayer; // y(n)
    VectorXd target; // d(n)
    SparseMatrix<double> W_in; //input-to-reservoir
    SparseMatrix<double> W; //reservoir
    SparseMatrix<double> W_out;
    MatrixXd stateCollection; 
    MatrixXd targetCollection; 
    VectorXd fIn(VectorXd v);
    inline VectorXd fOut(VectorXd v);
    void setInputTarget(Pattern p);
    inline VectorXd getOutput();
    void updateWeights();
    void train(); 
    void test();

protected:

};

ESN::ESN(){}
ESN::ESN(Dataset train, Dataset test, double sparsity, int washout, double spectralRadius, double leakingRate, double reg){
    trainDataset = train;
    testDataset = test;
    //p(W) < 1 <-> || W || < 1
    int input = trainDataset.getInputLayerSize();
    int reservoir = trainDataset.getHiddenLayerSize();
    int output = trainDataset.getOutputLayerSize();
    this-> washout = washout;
    this-> spectralRadius = spectralRadius;
    this -> leakingRate = leakingRate;
    this -> reg = reg;
    this -> sparsity = sparsity;
    inputLayer = VectorXd(input+1); 
    internalState = VectorXd(reservoir);
    outputLayer = VectorXd(output);
    target = VectorXd(output);
    W_in = SparseMatrix<double>(reservoir,1+input);
    W = SparseMatrix<double>(reservoir,reservoir); 
    W_out = SparseMatrix<double>(output, 1+input+reservoir); 
    stateCollection = MatrixXd(1+input+reservoir,trainDataset.getSize()-washout+1); 
    targetCollection = MatrixXd(output,trainDataset.getSize()-washout+1);
    W.setZero(); W_in.setZero(); W_out.setZero();
}



void ESN::createNetwork(){
    #ifdef PRINTS
    printf("Create Network ...\n");
    #endif
    createReservoir();  
    createInput();
}//end create network

void ESN::createReservoir() {
    vector< pair<int,int> > idx; vector< pair<int,int> > aux;
    random_device rdI; mt19937 genI(rdI()); uniform_int_distribution<> disI(0, W.rows()-1); 
    random_device rdJ; mt19937 genJ(rdJ()); uniform_int_distribution<> disJ(0, W.cols()-1);
    random_device rdR; mt19937 genR(rdR()); uniform_real_distribution<> disR(-0.5, 0.5); 
    random_device rdB; mt19937 genB(rdB()); uniform_real_distribution<> disB(-0.01,0.01); 
    
    vector< Eigen::Triplet<double> > tripletList;
    int percentNonZero = W.rows() * W.cols() * sparsity;   //20% on nonZero cells, from jaeger,lucosevicius, 2010
    tripletList.reserve(percentNonZero);
    for(int p =0; p<percentNonZero; p++){
        int i = disI(genI);
        int j = disJ(genJ);
        idx.push_back(make_pair(i,j));
        double value = 1e-7;
        do{ 
            if(j<W.cols()-1) value = disR(genR);
            else value = disB(genB);
        }while(value < 1e-6); 
        tripletList.push_back(Eigen::Triplet<double>(i,j,value)); 
    }
    W.setFromTriplets(tripletList.begin(), tripletList.end());
    SparseMatrix<double> W_0(W.rows(),W.cols());
    SelfAdjointEigenSolver<SparseMatrix<double> > es(W);
    W_0=W*(1/fabs(es.eigenvalues()[W.rows()-1]));//minimalESN normalize with 1.25, Jaeger 2002 with 1
    W = spectralRadius * W_0;
}
void ESN::createInput() {
    MatrixXd m = MatrixXd::Random(W_in.rows(),W_in.cols());
    m = m - (MatrixXd::Identity(m.rows(),m.cols()) / 2);
    W_in = m.sparseView();
}

inline void ESN::setReservoirSize(int n){ trainDataset.setHiddenUnit(n); }

VectorXd ESN::fIn(VectorXd v){
    VectorXd tv(v.size());
    for(int i=0; i<v.size(); i++)
        tv(i)=tanh(v(i));
    return tv;
}

inline VectorXd ESN::fOut(VectorXd v){ return fIn(v); }


void ESN::setInputTarget(Pattern p){
    vector<double> aux; aux.push_back(1);
    aux.insert(aux.begin()+1, p.input.begin(), p.input.end());
    Map<VectorXd> in(aux.data(),p.input.size()+1);
    Map<VectorXd> tar(p.output.data(),p.output.size());
    inputLayer = in; target = tar;
}

inline VectorXd ESN::getOutput(){ return outputLayer; }

void ESN::saveWeights(const string path){
    MatrixXd cpW = MatrixXd(W);
    ofstream f(path+"w.esn", ios::binary);
    MatrixXd::Index rows=cpW.rows(), cols=cpW.cols();
    f.write((char*) (&rows), sizeof(MatrixXd::Index));
    f.write((char*) (&cols), sizeof(MatrixXd::Index));
    f.write((char *) cpW.data(), rows * cols * sizeof(MatrixXd::Scalar));
    f.close(); 
    MatrixXd cpWin = MatrixXd(W_in);
    ofstream f1(path+"win.esn", ios::binary);
    rows=cpWin.rows(), cols=cpWin.cols();
    f1.write((char*) (&rows), sizeof(MatrixXd::Index));
    f1.write((char*) (&cols), sizeof(MatrixXd::Index));
    f1.write((char *) cpWin.data(), rows * cols * sizeof(MatrixXd::Scalar));
    f1.close(); 
    MatrixXd cpWout = MatrixXd(W_out);
    ofstream f2(path+"wout.esn", ios::binary);
    rows=cpWout.rows(), cols=cpWout.cols();
    f2.write((char*) (&rows), sizeof(MatrixXd::Index));
    f2.write((char*) (&cols), sizeof(MatrixXd::Index));
    f2.write((char *) cpWout.data(), rows * cols * sizeof(MatrixXd::Scalar));
    f2.close(); 
}

void ESN::restoreWeights(const string path){
    MatrixXd cpW;
    ifstream in(path+"w.esn",ios::in | std::ios::binary);
    MatrixXd::Index rows=0, cols=0;
    in.read((char*) (&rows), sizeof(MatrixXd::Index));
    in.read((char*) (&cols), sizeof(MatrixXd::Index));
    cpW.resize(rows, cols);
    in.read((char *) cpW.data(), rows*cols* sizeof(MatrixXd::Scalar));
    in.close();
    W = cpW.sparseView();
    MatrixXd cpWin;
    ifstream in1(path+"win.esn",ios::in | std::ios::binary);
    rows=0, cols=0;
    in1.read((char*) (&rows), sizeof(MatrixXd::Index));
    in1.read((char*) (&cols), sizeof(MatrixXd::Index));
    cpWin.resize(rows, cols);
    in1.read((char *) cpWin.data(), rows * cols * sizeof(MatrixXd::Scalar));
    in1.close();
    W_in= cpWin.sparseView();
    MatrixXd cpWout;
    ifstream in2(path+"wout.esn",ios::in | std::ios::binary);
    rows=0, cols=0;
    in2.read((char*) (&rows), sizeof(MatrixXd::Index));
    in2.read((char*) (&cols), sizeof(MatrixXd::Index));
    cpWout.resize(rows, cols);
    in2.read((char *) cpWout.data(), rows * cols * sizeof(MatrixXd::Scalar));
    in2.close();
    W_out= cpWout.sparseView();
}

void ESN::train(){   
    MSEtrain = 0; internalState.setZero(); target.setZero();
    for(int i=0; i<trainDataset.getSize(); i++) {
        double accumulateError = 0;
        VectorXd previousTarget = target;
        VectorXd previousState = internalState;
        setInputTarget(trainDataset.getPattern(i));    
        internalState/*n*/ = (1-leakingRate)*previousState + leakingRate * 
                fIn( W_in * inputLayer/*n*/ + W * previousState /*n-1*/ );
        if(i > washout){
            VectorXd stateRow(inputLayer.size()+internalState.size());
            stateRow << inputLayer, internalState;
            stateCollection.col(i-washout-1) = stateRow;
            targetCollection.col(i-washout-1) = target;
        } 
    } MSEtrain /= trainDataset.getSize();
}


void ESN::test(){ 
    MSEtest = 0; internalState.setZero(); target.setZero();
    for(int i=0; i<testDataset.getSize(); i++) {
        double accumulateError = 0;
        VectorXd previousTarget = target;
        VectorXd previousState = internalState;
        setInputTarget(testDataset.getPattern(i));
         internalState/*n*/ = (1-leakingRate)*previousState + leakingRate * 
                fIn( W_in * inputLayer/*n*/ + W * previousState /*n-1*/ );
        VectorXd concat(inputLayer.size()/*n*/ + internalState.size()/*n*/);
        concat << inputLayer, internalState;
        outputLayer/*n*/ = fOut(W_out * concat);
        for(int i=0; i<target.rows(); i++)
            accumulateError += outputLayer(i) - target(i);
        accumulateError /= target.rows();
        MSEtest += sqrt(pow(accumulateError,2));+
        plotTestVecX.push_back(inputLayer(1));
        plotTestVecY.push_back(outputLayer(0));
        plotTestVecT.push_back(target(0));
        //----------------------------
    } MSEtest /= testDataset.getSize();
}

void ESN::updateWeights(){  // By RIDGE REGRESSION
    int n = W.rows() + inputLayer.size(); 
    W_out = (targetCollection * stateCollection.transpose() * (stateCollection * stateCollection.transpose() + reg * MatrixXd::Identity(n,n) ).inverse()).sparseView();
    stateCollection.setZero(); targetCollection.setZero();
    
}

void ESN::compute(){ 
    #ifdef PRINTS
    cout << "Start computation... \n";
    #endif
    train();
    updateWeights();
    test();
    #ifdef PRINTS
    printf("Results: \nComputation stopped with MSEtest %f \n", MSEtest);
    #endif
}
void ESN::predict(Dataset blind, const char* output){ 
    Dataset blindDS = blind;
    outputLayer.setZero(); internalState.setZero(); target.setZero();
    ofstream f(output);
    for(int i=0; i<blindDS.getSize(); i++) {
        VectorXd previousTarget = target, previousState = internalState;
        setInputTarget(blindDS.getPattern(i));
         internalState/*n*/ = (1-leakingRate)*previousState + leakingRate * 
                fIn( W_in * inputLayer/*n*/ + W * previousState /*n-1*/ );
        VectorXd concat(inputLayer.size()/*n*/ + internalState.size()/*n*/);
        concat << inputLayer, internalState;
        outputLayer/*n*/ = fOut(W_out * concat);
        f << i << ",";
        f<< target(0) << ","; 
        f << outputLayer(0) << ",";
        f << "\n";
    }
    f.close();

}

inline double ESN::getError() { return MSEtest; }

void ESN::saveTestPlot(std::string filename) {
    if(Gnuplot::get_program_path()){
        Gnuplot gp("lines");
        gp.savetopng(filename);
        gp.set_xlabel("X");
        gp.set_ylabel("Y");
        gp.plot_xxy(plotTestVecX,plotTestVecY,plotTestVecT,"Results", "");
    } else {
        cerr << "Gnuplot is not installed in your machine. Plot not saved." << endl;
        cerr << "Don't worry!! Computation continues!" << endl;
    }
}

