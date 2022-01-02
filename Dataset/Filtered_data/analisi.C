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

array<double,2> compute_angles(double x_start, double y_start, double z_start,
                               double x_stop, double y_stop, double z_stop){
  // da ricontrollare, l'ho fatto in fretta
  double x = x_stop - x_start;
  double y = y_stop - y_start;
  double z = z_stop - z_start;
  array<double,2> angles;
  angles[0] = TMath::ATan((x*x + y*y)/z);// theta
  angles[1] = TMath::ATan(y/x);
  return angles;
}

void null(double shower[12][12][12][1]){
  for(int layers=0; layers<12; layers++){
    for(int num_z=0; num_z<12;num_z++){
      for(int num_y=0; num_y<12;num_y++){
        shower[layers][num_z][num_y][0] = 0 ;
      }
    }
  }
}

void formattazione_MVA(){
    //--------------------------------------------------------------------------
    // Funzione da usare solo sui dati grezzi !!!!
    //--------------------------------------------------------------------------
    ROOT::EnableImplicitMT(); // Tell ROOT you want to go parallel

    const char *input = "/mnt/c/Users/Daniele/Desktop/Magistrale/CMEPDA/shower_prova.root";
    const char *output = "data_MVA.root";
  // input, which is GEARS output
  	TChain *t = new TChain("t");
  	t->Add(input);
  	int nStepPoints; // number of Geant4 step points
  	t->SetBranchAddress("n",&nStepPoints);
  	// parameters of step points
    vector<double> *x_=0, *y_=0, *z_=0, *de_=0, *k_=0, *et_=0;
  	vector<int> *pid_=0, *pdg_=0, *vlm_=0; // copy number of a Geant4 volume
  	TBranch *bx, *by, *bz, *be, *bv, *bk, *bp, *bet, *bpid;
  	t->SetBranchAddress("x",&x_, &bx); // global x
  	t->SetBranchAddress("y",&y_, &by); // global y
  	t->SetBranchAddress("z",&z_, &bz); // global z
  	t->SetBranchAddress("de",&de_, &be); // energy deposition
    t->SetBranchAddress("k",&k_, &bk); // kinetic energy
    t->SetBranchAddress("et",&et_, &bet); // total energy deposition
  	t->SetBranchAddress("vlm",&vlm_, &bv); // sensitive volume
  	t->SetBranchAddress("pdg",&pdg_, &bp); //particle
    t->SetBranchAddress("pid",&pid_, &bpid); //particle parent

   	// output
  	TFile *file = new TFile(output, "recreate");
  	TTree *tree = new TTree("h","ttree");
  	int evt; // id of event from Geant4 simulation
    int primary;
    double theta, phi, en_in;
    tree->Branch("evt",&evt,"evt/I");
    tree->Branch("primary",&primary, "primary/I");
    tree->Branch("en_in", &en_in, "en_in/D");
    tree->Branch("theta", &theta, "theta/D");
    tree->Branch("phi", &phi, "phi/D");
    double shower[12][12][12][1];
    // 3D vector: first index = layer; second index = z-coordinate of unit cell;
    // third index = y-coordinate of unit cell
    tree->Branch("shower", &shower, "shower[12][12][12][1]/D");

    double x_start, y_start, z_start;
    double x_stop, y_stop, z_stop;
    array<double,2> angles;
    Bool_t check_en, check_start, check_stop;
    double check_en_vol=0;

  	// main loop
  	int nevt = t->GetEntries(); // total number of events simulated
  	cout<<nevt<<" events to be processed"<<endl;

    for (evt=0; evt<nevt; evt++) {
      if (evt%50==0) cout<<evt<<" events processed"<<endl;
      t->GetEntry(evt); // get information from input tree
      check_en = kFALSE; check_start = kFALSE; check_stop = kFALSE;
      null(shower);
      for(int layers=1; layers<13; layers++){
        int j=0;
        for(int num_z=1; num_z<13;num_z++){
          for(int num_y=1; num_y<13;num_y++){
            j++;
            for (int i=0; i<nStepPoints; i++) {
              if(!check_en){
                if(vlm_->at(i) == 1 && k_->at(i)>100000){
                  en_in = k_->at(i);
                  if(pdg_->at(i)==22) primary = 0;
                  if(pdg_->at(i)==11) primary = 1;
                  if(pdg_->at(i)==-11) primary = -1;
                  check_en=kTRUE;
                }
              }
              if(!check_start){
                if(pid_->at(i)==0 && x_->at(i)<-180){
                  x_start = x_->at(i); y_start = y_->at(i); z_start = z_->at(i);
                  check_start = kTRUE;
                }
              }
              if(!check_stop){
                if(pid_->at(i)==0 && x_->at(i)>-30 ){//&& x_->at(i)<-5
                  x_stop = x_->at(i); y_stop = y_->at(i); z_stop = z_->at(i);
                  check_stop = kTRUE;
                  angles = compute_angles(x_start,y_start,z_start,x_stop,y_stop,z_stop);
                  theta = angles[0]; phi = angles[1];
                }
              }
              if(vlm_->at(i)==(layers)*1000 + j){
                  shower[layers-1][num_z-1][num_y-1][0]+=de_->at(i);
              }
            }
          }
        }
      }
      for(int layers=0; layers<12; layers++){
        for(int num_z=0; num_z<12;num_z++){
          for(int num_y=0; num_y<12;num_y++){
            if(shower[layers][num_z][num_y][0]==0){
              shower[layers][num_z][num_y][0] =-5.;
            }
            else{
              shower[layers][num_z][num_y][0] = TMath::Log10(shower[layers][num_z][num_y][0]) ;
            }
          }
        }
      }
      tree->Fill();
    }

  	// save the output tree
    tree->Write("", TObject::kWriteDelete); // write tree, then delete previous
    file->Close(); // close output file
}

TH2D *set_hist_layer(const int LAYER, double shower[12][12][12][1]){
  char label[50];
  sprintf(label, "layer %d;y[mm];z[mm]", LAYER);

  TH2D *layer_x = new TH2D("",label, 12,-200,200,12,-200,200);
  for(int num_z=0; num_z<12;num_z++){
    for(int num_y=0; num_y<12;num_y++){
      std::cout<<shower[LAYER-1][num_z][num_y][0]<<std::endl;
      layer_x->SetBinContent(num_z, num_y, shower[LAYER-1][num_z][num_y][0]);
    }
  }
  return layer_x;
}

void do_stuff(TCanvas *c, int index, TH2D *hist){
  c->cd(index);
  gPad->SetLogz();
  hist->SetContour(99);
  hist->SetMaximum(240000); hist->SetMinimum(1);
  hist->Draw("COLZ");
}

void event_display(int const evento=0, Bool_t show_display = kTRUE){

  ROOT::EnableImplicitMT(); // Tell ROOT you want to go parallel

  const char *input="dati_MVA.root";
  TChain *h = new TChain("h");
  h->Add(input);

  double shower[12][12][12][1];
  TBranch *b_shower, *b_en_in;
  double en_in, en_misurata=0;
  h->SetBranchAddress("en_in", &en_in, &b_en_in);
  h->SetBranchAddress("shower", shower, &b_shower);

  h->GetEntry(evento);

  TH2D *layer1 = new TH2D(); TH2D *layer2 = new TH2D(); TH2D *layer3 = new TH2D();
  TH2D *layer4 = new TH2D(); TH2D *layer5 = new TH2D(); TH2D *layer6 = new TH2D();
  TH2D *layer7 = new TH2D(); TH2D *layer8 = new TH2D(); TH2D *layer9 = new TH2D();
  TH2D *layer10 = new TH2D(); TH2D *layer11 = new TH2D(); TH2D *layer12 = new TH2D();

  layer1 = set_hist_layer(1, shower); layer2 = set_hist_layer(2, shower);layer3 = set_hist_layer(3, shower);
  layer4 = set_hist_layer(4, shower); layer5 = set_hist_layer(5, shower);layer6 = set_hist_layer(6, shower);
  layer7 = set_hist_layer(7, shower); layer8 = set_hist_layer(8, shower);layer9 = set_hist_layer(9, shower);
  layer10 = set_hist_layer(10, shower); layer11 = set_hist_layer(11, shower);layer12 = set_hist_layer(12, shower);

  if(show_display){
    TCanvas *c = new TCanvas("","",1200,800);
    c->Divide(4,3);
    gStyle->SetOptStat(kFALSE);
    //gStyle->SetPalette(52, 0);
    do_stuff(c, 1, layer1);
    do_stuff(c, 2, layer2);
    do_stuff(c, 3, layer3);
    do_stuff(c, 4, layer4);
    do_stuff(c, 5, layer5);
    do_stuff(c, 6, layer6);
    do_stuff(c, 7, layer7);
    do_stuff(c, 8, layer8);
    do_stuff(c, 9, layer9);
    do_stuff(c, 10, layer10);
    do_stuff(c, 11, layer11);
    do_stuff(c, 12, layer12);
  }

  for(int layers=0; layers<12; layers++){
    for(int num_z=0; num_z<12;num_z++){
      for(int num_y=0; num_y<12;num_y++){
        en_misurata += shower[layers][num_z][num_y][0] ;
      }
    }
  }

  std::cout<<"Energia iniziale:\t"<<en_in<<"keV"<<std::endl;
  std::cout<<"Energia misurata:\t"<<en_misurata<<"keV"<<std::endl;

}
