#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TFrame.h"
#include "TBenchmark.h"
#include "TString.h"
#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TROOT.h"
#include "TError.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "TPaveText.h"
#include "math.h"
#include "TThread.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

const char* path_to_Geant_data = "../dataset/filtered_data/data_MVA.root";
const char* path_to_GAN_data   = "../dataset/GAN_data/data_GAN.root";

int const NUMBER_OF_LAYERS = 12;
int const NUMBER_OF_PIXEL_Z = 12;
int const NUMBER_OF_PIXEL_Y = 12;

TH2D *set_hist_layer(const int LAYER, double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1]){
  /**
  Fills an histogram of the layer #LAYER using the contents of shower 3Dx1 vector
  */
  char label[50];
  sprintf(label, "layer %d;y[mm];z[mm]", LAYER);

  TH2D *layer_x = new TH2D("",label, 11,-200,200,11,-200,200);
  for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
    for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
      layer_x->SetBinContent(num_z, num_y, shower[LAYER-1][num_z][num_y][0]);
      //std::cout<< LAYER-1 <<"\t"<< num_z <<"\t"<< num_y <<"\t"<< layer_x->GetBinContent(num_z, num_y)<<std::endl;
    }
  }
  return layer_x;
}

void single_data_analysis(const char* path_to_file=path_to_Geant_data){
  /**
  Analyse data
  */
  TChain *h = new TChain("h");
  h->Add(path_to_file);

  double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1];
  TBranch *b_shower, *b_en_in, *b_pid, *b_en_mis;
  double en_in, en_mis;
  int pid;
  h->SetBranchAddress("primary", &pid, &b_pid);
  h->SetBranchAddress("en_in", &en_in, &b_en_in);
  h->SetBranchAddress("en_mis", &en_mis, &b_en_mis);
  h->SetBranchAddress("shower", shower, &b_shower);

  int nevt = h->GetEntries();
  h->GetEntry(1000);


  for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
    for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
      for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
        shower[layers][num_z][num_y][0] = TMath::Power(10,shower[layers][num_z][num_y][0]) ;
      }
    }
  }

  vector<TH2D*> layer(NUMBER_OF_LAYERS);
  double mean_z, mean_y; // z=x , y=y


  for(int i=0; i<NUMBER_OF_LAYERS;i++){
    layer[i] = set_hist_layer(i+1, shower);
    mean_z = layer[i]->ProjectionX()->GetMean();
    mean_y = layer[i]->ProjectionY()->GetMean();
  }

  layer[5]->Draw("zcolor");



}
