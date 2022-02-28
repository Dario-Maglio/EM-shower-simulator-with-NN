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

const char* path_to_Geant_data_1 = "../dataset/filtered_data/data_MVA_24pixel_parte1.root";
const char* path_to_Geant_data_2 = "../dataset/filtered_data/data_MVA_24pixel_parte2.root";

const char* path_to_GAN_data   = "../dataset/gan_data/data_GAN.root";

int const NUMBER_OF_LAYERS = 12;
int const NUMBER_OF_PIXEL_Z = 25;
int const NUMBER_OF_PIXEL_Y = 25;

double const EN_NORM = 6.7404;

TChain* build_tree(int num_of_path, ...){
  /**
  Returns ttree filled with GAN or GEANT dataset
  */
  va_list path_list;
  va_start(path_list, num_of_path);
  TChain *h = new TChain("h");

  for (int i = 0; i < num_of_path; i++) {
    h->Add(va_arg(path_list, const char*));
    // cout<<"ok build"<<endl;
  }
  va_end(path_list);

  return h;
}


void do_stuff(TCanvas *c, int index, TH2D *hist){
  /**
  Plots the histogram inside a canvas
  */
  c->cd(index);
  // gPad->SetLogz();
  hist->SetContour(199);
  //hist->SetMaximum(6.7404);
  //hist->SetMinimum(-1);
  hist->Draw("COLZ");
}

TH2D *construct_hist_vector(int bins_x, double min_x, double max_x,
                            int bins_y, double min_y, double max_y){
  /**
  Construct histogram vector member
  */

  TH2D *layer_x = new TH2D("","", bins_x,min_x,max_x,bins_y, min_y,max_y);
  return layer_x;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

vector<TH2D*> mean_layers(const char* path_to_file="GEANT"){
  /**
  Analyse data: mean energy deposition per layer, in normalized log-scale
  */
  TChain *h = new TChain("h");
  TChain *tmp1 = new TChain("h");

  if(strncmp (path_to_file,"GEANT",2) == 0){
    h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);
  }
  else if(strncmp (path_to_file,"GAN",2) == 0){
    h = build_tree(1, path_to_GAN_data);
  }

  double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1];
  TBranch *b_shower, *b_en_in, *b_pid, *b_en_mis;
  double en_in, en_mis;
  int pid;
  h->SetBranchAddress("primary", &pid, &b_pid);
  h->SetBranchAddress("en_in", &en_in, &b_en_in);
  h->SetBranchAddress("en_mis", &en_mis, &b_en_mis);
  h->SetBranchAddress("shower", shower, &b_shower);
  vector<TH2D*> mean_layer(NUMBER_OF_LAYERS);

  for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
    mean_layer[layers] = construct_hist_vector(NUMBER_OF_PIXEL_Z,-200.,200.,
                                               NUMBER_OF_PIXEL_Y,-200.,200.);
  }

  int nevt = h->GetEntries();
  int nevt_to_analyze = nevt/1;

  for(int evt=0; evt<nevt_to_analyze; evt++){
    if (evt%50==0){
      cout<<evt<<" events processed "<<endl;
    }
    h->GetEntry(evt);
    for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
      for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
        for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
          shower[layers][num_z][num_y][0] = TMath::Power(10,shower[layers][num_z][num_y][0]*EN_NORM)/1E6 ;
          mean_layer[layers]->SetBinContent(num_z+1, num_y+1,
            mean_layer[layers]->GetBinContent(num_z+1,num_y+1)+shower[layers][num_z][num_y][0] );
        }
      }
    }
  }

  TCanvas *c = new TCanvas("","",1600,1000);
  int x_div = int(TMath::Sqrt(NUMBER_OF_LAYERS))+1;
  int y_div;
  if(x_div*(x_div-1)==NUMBER_OF_LAYERS){y_div=x_div-1;}
  else {y_div = x_div;}
  c->Divide(x_div,y_div);
  gStyle->SetOptStat(kFALSE);
  char label[50];
  for(int i=0; i<NUMBER_OF_LAYERS; i++){
    sprintf(label, "layer %d;y[mm];z[mm]", i+1);
    mean_layer[i]->SetTitle(label);
    mean_layer[i]->Scale(1./nevt_to_analyze);
    do_stuff(c, i+1, mean_layer[i]);
  }

  return mean_layer;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

vector<TH2D*> mean_en_deposition_per_layer_per_particle(const char* path_to_file="GEANT"){
  /**
  Analyze data: mean energy deposition per layer and per particle, in normalized log-scale
  */
  TChain *h = new TChain("h");

  if(strncmp (path_to_file,"GEANT",2) == 0){
    h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);
  }
  else if(strncmp (path_to_file,"GAN",2) == 0){
    h = build_tree(1, path_to_GAN_data);
  }

  double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1];
  TBranch *b_shower, *b_en_in, *b_pid, *b_en_mis;
  double en_in, en_mis;
  int pid;
  h->SetBranchAddress("primary", &pid, &b_pid);
  h->SetBranchAddress("en_in", &en_in, &b_en_in);
  h->SetBranchAddress("en_mis", &en_mis, &b_en_mis);
  h->SetBranchAddress("shower", shower, &b_shower);

  double en_inside_layer[NUMBER_OF_LAYERS];
  vector<TH2D*> hist_mean_lateral_layer(2);
  for(int pid=0; pid<2; pid++){
    hist_mean_lateral_layer[pid] = construct_hist_vector(12,0.,12.,
                                                    29,1.,30.);
  }
  vector<TH2D*> hist_mean(2);
  for(int pid=0; pid<2; pid++){
    hist_mean[pid] = construct_hist_vector(12,0.,12.,
                                            29,1.,30.);
  }

  int nevt = h->GetEntries();
  int nevt_to_analyze = nevt/1;
  int count_photons =0;
  int count_leptons =0;

  for(int evt=0; evt<nevt_to_analyze; evt++){
    if (evt%50==0){
      cout<<evt<<" events processed "<<endl;
    }
    h->GetEntry(evt);
    if(strncmp (path_to_file,"GAN",2) == 0){
      // cout<<pid<<endl;
      pid--;
    }
    if(TMath::Abs(pid)==1){
      count_leptons++;
    }
    if(TMath::Abs(pid)==0){
      count_photons++;
    }
    for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
      en_inside_layer[layers] = 0;
      for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
        for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
          shower[layers][num_z][num_y][0] = TMath::Power(10,shower[layers][num_z][num_y][0]*EN_NORM)/1E6 ;
          en_inside_layer[layers] += shower[layers][num_z][num_y][0];
        }
      }
      // cout<<en_in/1E6<<"\t"<<en_inside_layer[layers]<<endl;
      hist_mean_lateral_layer[TMath::Abs(pid)]->SetBinContent(layers+1, en_in/1.E6, en_inside_layer[layers]);
    }
    if(TMath::Abs(pid)==1){
      hist_mean[1] ->Add(hist_mean_lateral_layer[1]);
    }

    if(TMath::Abs(pid)==0){
      hist_mean[0] ->Add( hist_mean_lateral_layer[0]);
    }
  }

  TCanvas *c = new TCanvas("","",1200,500);
  c->Divide(2,1);
  gStyle->SetOptStat(kFALSE);
  for(int i=0; i<2; i++){
    switch (i) {
      case 0: {
              hist_mean[i]->Scale(1./count_photons);
              hist_mean[i]->SetTitle("Photons; Layer; Initial energy [GeV]");
              break;}
      case 1: {
              hist_mean[i]->Scale(1./count_leptons);
              hist_mean[i]->SetTitle("Positrons and Electrons; Layer; Initial energy [GeV]");
              break;}
    }
    do_stuff(c, i+1, hist_mean[i]);
    hist_mean[i]->SetMaximum(8);
  }

  return hist_mean;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Non so quanto ha senso questa cosa che ho fatto

// int const nevt_to_analyze = 10000;
//
// void quantile_per_layer(double_t quantile[nevt_to_analyze][NUMBER_OF_LAYERS][3], const char* path_to_file="GEANT", int const user_pid=-1){
//
//   TChain *h = new TChain("h");
//
//   if(strncmp (path_to_file,"GEANT",2) == 0){
//     h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);
//   }
//   else if(strncmp (path_to_file,"GAN",2) == 0){
//     h = build_tree(1, path_to_GAN_data);
//   }
//
//   double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1];
//   TBranch *b_shower, *b_en_in, *b_pid, *b_en_mis;
//   double en_in, en_mis;
//   int pid;
//   h->SetBranchAddress("primary", &pid, &b_pid);
//   h->SetBranchAddress("en_in", &en_in, &b_en_in);
//   h->SetBranchAddress("en_mis", &en_mis, &b_en_mis);
//   h->SetBranchAddress("shower", shower, &b_shower);
//
//   vector<TH2D*> layer(NUMBER_OF_LAYERS), centred_layers(NUMBER_OF_LAYERS);
//   for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
//     layer[layers] = construct_hist_vector(11,1.,12.,11,1.,12.);
//     centred_layers[layers] = construct_hist_vector(11,1.,12.,11,1.,12.);
//   }
//
//   int nevt = h->GetEntries();
//   //int nevt_to_analyze = nevt/1;
//   double_t quantile_level[3]={0.62, 0.95, 0.99};
//   int mean_x, mean_y;
//
//   for(int evt=0; evt<nevt_to_analyze; evt++){
//     if (evt%50==0){
//       cout<<evt<<" events processed "<<endl;
//     }
//     h->GetEntry(evt);
//
//     if(pid==user_pid){
//       for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
//         for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
//           for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
//             layer[layers]->SetBinContent(num_z,num_y,
//                          TMath::Power(10,shower[layers][num_z][num_y][0])) ;
//           }
//         }
//       }
//       mean_x = int(layer[0]->GetMean(1));
//       mean_y = int(layer[0]->GetMean(2));
//       for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
//         centred_layers[layers]->SetBins(NUMBER_OF_PIXEL_Z -1 - mean_x,0.,NUMBER_OF_PIXEL_Z - mean_x,
//           NUMBER_OF_PIXEL_Y-1-mean_y,0., NUMBER_OF_PIXEL_Y-mean_y);
//         for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z-mean_x;num_z++){
//           for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y-mean_y;num_y++){
//
//             if(mean_x-num_z>=0 && mean_y-num_y && mean_x+num_z< NUMBER_OF_PIXEL_Z && mean_y+num_y<NUMBER_OF_PIXEL_Y){
//               centred_layers[layers]->SetBinContent(num_z,num_y,
//                 (layer[layers]->GetBinContent(mean_x-num_z,mean_y-num_y)+
//                   layer[layers]->GetBinContent(mean_x+num_z,mean_y+num_y)) /2.  ) ;
//             }
//
//             else if(mean_x-num_z>=0 && mean_y-num_y && (mean_x+num_z> NUMBER_OF_PIXEL_Z || mean_y+num_y>NUMBER_OF_PIXEL_Y) ){
//               centred_layers[layers]->SetBinContent(num_z,num_y,
//                 layer[layers]->GetBinContent(mean_x-num_z,mean_y-num_y));
//             }
//             else if(mean_x+num_z< NUMBER_OF_PIXEL_Z && mean_y+num_y<NUMBER_OF_PIXEL_Y && (mean_x-num_z>=0 || mean_y-num_y) ){
//               centred_layers[layers]->SetBinContent(num_z,num_y,
//                 layer[layers]->GetBinContent(mean_x+num_z,mean_y+num_y));
//             }
//
//             else centred_layers[layers]->SetBinContent(num_z,num_y,0.0001);
//           }
//         }
//         centred_layers[layers]->ProjectionX()->GetQuantiles(3,quantile[evt][layers],quantile_level);
//         //cout<<"Evento "<<evt<<" ; Layer "<<layers<<"\t\t"<<quantile[evt][layers][0]<<"\t"<<quantile[evt][layers][1]<<"\t"<<quantile[evt][layers][2]<<endl;
//       }
//     }
//
//   }
//
//   /*TCanvas *c = new TCanvas("","",1200,800);
//   c->Divide(4,3);
//
//   for(int i=0; i<NUMBER_OF_LAYERS; i++){
//     do_stuff(c, i+1, centred_layers[i]);
//   }*/
//
// }
//
// void quantiles(){
//   double_t quantile[nevt_to_analyze][NUMBER_OF_LAYERS][3];
//   const char * path_to_file = "GEANT";
//   quantile_per_layer(quantile,path_to_file, -1); //Positrons
//
//   TH2D *quantile_062 = new TH2D("","@ 0.62",11,1,12,100,0,8);
//   TH2D *quantile_095 = new TH2D("","@ 0.95",11,1,12,100,0,8);
//   TH2D *quantile_099 = new TH2D("","@ 0.99",11,1,12,100,0,8);
//
//   TChain *h = new TChain("h");
//
//   if(strncmp (path_to_file,"GEANT",2) == 0){
//     h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);
//   }
//   else if(strncmp (path_to_file,"GAN",2) == 0){
//     h = build_tree(1, path_to_GAN_data);
//
//   }
//
//   TBranch *b_pid;
//   int pid;
//   h->SetBranchAddress("primary", &pid, &b_pid);
//
//   for(int i=0; i<nevt_to_analyze; i++){
//     h->GetEntry(i);
//     for(int layer=0; layer<NUMBER_OF_LAYERS; layer++){
//       if(pid==-1){
//         //cout<<i<<"\t"<<layer<<"\t"<<quantile[i][layer][0]<<endl;
//         quantile_062->Fill(layer,quantile[i][layer][0])  ;
//         quantile_095->Fill(layer,quantile[i][layer][1])  ;
//         quantile_099->Fill(layer,quantile[i][layer][2])  ;
//       }
//     }
//   }
//   TCanvas *c = new TCanvas("","",1200,500);
//   c->Divide(3,1);
//
//   c->cd(1);
//   quantile_062->Draw("zcolor");
//   gPad->SetLogz();
//
//   c->cd(2);
//   quantile_095->Draw("zcolor");
//   gPad->SetLogz();
//
//   c->cd(3);
//   quantile_099->Draw("zcolor");
//   gPad->SetLogz();
//
// }
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

void pdf_en_deposited(){

  TChain *h = new TChain("h");
  TChain *t = new TChain("h");

  h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);

  t = build_tree(1, path_to_GAN_data);

  TBranch *b_en_in_geant, *b_en_mis_geant;
  double en_in_geant, en_mis_geant;
  h->SetBranchAddress("en_in", &en_in_geant, &b_en_in_geant);
  h->SetBranchAddress("en_mis", &en_mis_geant, &b_en_mis_geant);

  TBranch *b_en_in_gan, *b_en_mis_gan;
  double en_in_gan, en_mis_gan;
  t->SetBranchAddress("en_in", &en_in_gan, &b_en_in_gan);
  t->SetBranchAddress("en_mis", &en_mis_gan, &b_en_mis_gan);

  TGraph *graph_geant = new TGraph();
  TGraph *graph_gan = new TGraph();

  for(int evt=0; evt<h->GetEntries(); evt++){
    h->GetEntry(evt);
    graph_geant->AddPoint(en_in_geant/1E6,en_mis_geant/1E6);
  }

  for(int evt=0; evt<t->GetEntries(); evt++){
    t->GetEntry(evt);
    graph_gan->AddPoint(en_in_gan/1E6,en_mis_gan/1E6);
  }

  TF1 *identity = new TF1("en_in = en_mis", "x", 0.,30.);

  graph_geant->Draw("AP");
  graph_geant->SetNameTitle("GEANT", "Energia iniziale vs misurata;Enegia iniziale [GeV];Energia misurata [GeV]");
  graph_geant->SetMinimum(0.);
  graph_geant->SetMarkerColor(kBlack);
  graph_geant->SetMarkerStyle(20);
  graph_geant->SetMarkerSize(0.5);

  graph_gan->Draw("P");
  graph_gan->SetName("GAN");
  graph_gan->SetMarkerColor(kBlue);
  graph_gan->SetMarkerStyle(20);
  graph_gan->SetMarkerSize(0.5);

  identity -> Draw("same");
  gPad->SetGrid();
  TLegend *legend = new TLegend(0.1,0.7,0.48,0.9);
  legend->AddEntry(identity,"en_in = en_mis","l");
  legend->AddEntry(graph_geant,"GEANT","p");
  legend->AddEntry(graph_gan,"GAN","p");
  legend->Draw();
}

void shower_depth(const char* path_to_file="GEANT"){

  TChain *h = new TChain("h");

  if(strncmp (path_to_file,"GEANT",2) == 0){
    h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);
  }
  else if(strncmp (path_to_file,"GAN",2) == 0){
    h = build_tree(1, path_to_GAN_data);
  }

  double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1];
  TBranch *b_shower, *b_en_in, *b_pid, *b_en_mis;
  double en_in, en_mis;
  int pid;
  h->SetBranchAddress("primary", &pid, &b_pid);
  h->SetBranchAddress("en_in", &en_in, &b_en_in);
  h->SetBranchAddress("en_mis", &en_mis, &b_en_mis);
  h->SetBranchAddress("shower", shower, &b_shower);

  int nevt = h->GetEntries();
  int nevt_to_analyze = nevt/1;

  double en_inside_layer, en_depth;
  vector<TH2D*> hist_mean_lateral_layer(2);
  for(int pid=0; pid<2; pid++){
    hist_mean_lateral_layer[pid] = construct_hist_vector(11,1.,12.,
                                                    29,1.,30.);
  }

  TH1D *hist_0 = new TH1D("","",29,1.,30.);
  TH1D *hist_1 = new TH1D("","",29,1.,30.);

  TH1D *hist_mean_0 = new TH1D("Photons","Shower Depth; Initial energy [GeV]; Depth [#layer]",29,1.,30.);
  TH1D *hist_mean_1 = new TH1D("Electrons and Positrons","Shower Depth; Initial energy [GeV]; Depth [#layer]",29,1.,30.);

  int count_photon =0;
  int count_electron =0;

  for(int evt=0; evt<nevt_to_analyze; evt++){
    if (evt%50==0){
      cout<<evt<<" events processed "<<endl;
    }
    h->GetEntry(evt);
    for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
      en_inside_layer = 0;
      for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
        for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
          shower[layers][num_z][num_y][0] = TMath::Power(10,shower[layers][num_z][num_y][0]*EN_NORM) ;
          en_inside_layer += shower[layers][num_z][num_y][0]/1E6;
        }
      }
      if(pid == 0){
        hist_mean_lateral_layer[0]->SetBinContent(layers+1, en_in/1.E6, en_inside_layer*layers/(en_mis/1E6)  ) ;
      }
      if(pid == 1 || pid == -1){
        hist_mean_lateral_layer[1]->SetBinContent(layers+1, en_in/1.E6, en_inside_layer*layers/(en_mis/1E6)  ) ;
      }

    }
    if(TMath::Abs(pid)==1){
      count_electron++;
      hist_1 = hist_mean_lateral_layer[1]->ProjectionY();
      hist_mean_1 ->Add(hist_1);
    }

    if(TMath::Abs(pid)==0){
      count_photon++;
      hist_0 = hist_mean_lateral_layer[0]->ProjectionY();
      hist_mean_0 ->Add(hist_0);
    }
  }

  hist_mean_0->Scale(1./count_photon);
  hist_mean_0->Draw("HIST");
  hist_mean_0->SetLineWidth(2);
  hist_mean_0->SetLineColor(kBlue);

  hist_mean_1->Scale(1./count_electron);
  hist_mean_1->Draw("HIST same");
  hist_mean_1->SetLineWidth(2);
  hist_mean_1->SetLineColor(kRed);

  gStyle->SetOptStat(kFALSE);
  gPad->SetGrid();
  TLegend *legend = new TLegend(0.1,0.7,0.38,0.9);
  legend -> AddEntry(hist_mean_0, "Photons", "l");
  legend -> AddEntry(hist_mean_1, "e^{#pm}", "l");
  legend -> Draw();

}

void shower_depth_width(const char* path_to_file="GEANT"){

  TChain *h = new TChain("h");

  if(strncmp (path_to_file,"GEANT",2) == 0){
    h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);
  }
  else if(strncmp (path_to_file,"GAN",2) == 0){
    h = build_tree(1, path_to_GAN_data);
  }

  double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1];
  TBranch *b_shower, *b_en_in, *b_pid, *b_en_mis;
  double en_in, en_mis;
  int pid;
  h->SetBranchAddress("primary", &pid, &b_pid);
  h->SetBranchAddress("en_in", &en_in, &b_en_in);
  h->SetBranchAddress("en_mis", &en_mis, &b_en_mis);
  h->SetBranchAddress("shower", shower, &b_shower);

  int nevt = h->GetEntries();
  int nevt_to_analyze = nevt/1;

  double en_inside_layer[NUMBER_OF_LAYERS], en_depth;
  double x,x2,x2_;
  vector<TH2D*> hist_sigma_layer(2);
  for(int pid=0; pid<2; pid++){
    hist_sigma_layer[pid] = construct_hist_vector(11,1.,12.,
                                                    29,1.,30.);
  }


  TH1D *hist_0 = new TH1D("","",29,1.,30.);
  TH1D *hist_1 = new TH1D("","",29,1.,30.);

  TH1D *hist_mean_0 = new TH1D("Photons","Shower Depth Widht; Initial energy [GeV]; Width [#layer]",29,1.,30.);
  TH1D *hist_mean_1 = new TH1D("Electrons and Positrons","Shower Depth Widht; Initial energy [GeV]; Widht [#layer]",29,1.,30.);

  int count_photon =0;
  int count_electron =0;

  for(int evt=0; evt<nevt_to_analyze; evt++){
    if (evt%50==0){
      cout<<evt<<" events processed "<<endl;
    }
    h->GetEntry(evt);
    for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
      en_inside_layer[layers] = 0;
      x=0;
      x2=0;
      x2_=0;
      for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
        for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
          shower[layers][num_z][num_y][0] = TMath::Power(10,shower[layers][num_z][num_y][0]*EN_NORM) ;
          en_inside_layer[layers] += shower[layers][num_z][num_y][0]/1E6;
        }
      }
    }

    if(TMath::Abs(pid)==1){
      count_electron++;
      for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
        x += TMath::Power(layers,2) * en_inside_layer[layers] /(en_mis/1.E6) ;//x2 += TMath::Power(layers*en_inside_layer[layers] /(en_mis/1.E6),2);
        x2_ += layers*en_inside_layer[layers] /(en_mis/1.E6) ;
      }
      x2 = TMath::Power(x2_, 2);
      hist_1->SetBinContent(en_in/1.E6,  TMath::Sqrt( TMath::Abs(x-x2) ) );
      hist_mean_1 ->Add(hist_1);
    }

    if(TMath::Abs(pid)==0){
      count_photon++;
      for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
        x += TMath::Power(layers,2) * en_inside_layer[layers] /(en_mis/1.E6) ;
        x2_ += layers*en_inside_layer[layers] /(en_mis/1.E6) ;
      }
      x2 = TMath::Power(x2_, 2);
      hist_0->SetBinContent(en_in/1.E6,  TMath::Sqrt( TMath::Abs(x-x2) ) );
      hist_mean_0 ->Add(hist_0);
    }
  }

  hist_mean_0->Scale(1./count_photon);
  hist_mean_0->Draw("HIST");
  hist_mean_0->SetLineWidth(2);
  hist_mean_0->SetLineColor(kBlue);

  hist_mean_1->Scale(1./count_electron);
  hist_mean_1->Draw("HIST same");
  hist_mean_1->SetLineWidth(2);
  hist_mean_1->SetLineColor(kRed);

  gStyle->SetOptStat(kFALSE);
  gPad->SetGrid();
  TLegend *legend = new TLegend(0.1,0.7,0.38,0.9);
  legend -> AddEntry(hist_mean_0, "Photons", "l");
  legend -> AddEntry(hist_mean_1, "e^{#pm}", "l");
  legend -> Draw();

}

void lateral_width(const char* path_to_file="GEANT"){

  TChain *h = new TChain("h");

  if(strncmp (path_to_file,"GEANT",2) == 0){
    h = build_tree(2, path_to_Geant_data_1, path_to_Geant_data_2);
  }
  else if(strncmp (path_to_file,"GAN",2) == 0){
    h = build_tree(1, path_to_GAN_data);
  }

  double shower[NUMBER_OF_LAYERS][NUMBER_OF_PIXEL_Z][NUMBER_OF_PIXEL_Y][1];
  TBranch *b_shower, *b_en_in, *b_pid, *b_en_mis;
  double en_in, en_mis;
  int pid;
  h->SetBranchAddress("primary", &pid, &b_pid);
  h->SetBranchAddress("en_in", &en_in, &b_en_in);
  h->SetBranchAddress("en_mis", &en_mis, &b_en_mis);
  h->SetBranchAddress("shower", shower, &b_shower);

  int nevt = h->GetEntries();
  int nevt_to_analyze = nevt/1;

  double en_inside_layer, en_depth;
  vector<TH2D*> hist_mean_lateral_layer(2);
  for(int pid=0; pid<2; pid++){
    hist_mean_lateral_layer[pid] = construct_hist_vector(11,1.,12.,
                                                    29,1.,30.);
  }

  double x_photons[NUMBER_OF_LAYERS];
  double x_leptons[NUMBER_OF_LAYERS];
  double x2_photons[NUMBER_OF_LAYERS];
  double x2_leptons[NUMBER_OF_LAYERS];


  TH1D *hist_0 = new TH1D("","",29,1.,30.);
  TH1D *hist_1 = new TH1D("","",29,1.,30.);

  TH1D *hist_mean_0 = new TH1D("Photons","Lateral width averaged over layers; Initial energy [GeV]; Width [#bins]",29,1.,30.);
  TH1D *hist_mean_1 = new TH1D("Electrons and Positrons","Lateral width averaged over layers; Initial energy [GeV]; Width [#bins]",29,1.,30.);

  int count_photon =0;
  int count_electron =0;

  for(int evt=0; evt<nevt_to_analyze; evt++){
    if (evt%50==0){
      cout<<evt<<" events processed "<<endl;
    }
    h->GetEntry(evt);
    for(int layers=0; layers<NUMBER_OF_LAYERS; layers++){
      en_inside_layer = 0;
      x_photons[layers]=0;
      x2_photons[layers]=0;
      x_leptons[layers]=0;
      x2_leptons[layers]=0;
      for(int num_z=0; num_z<NUMBER_OF_PIXEL_Z;num_z++){
        for(int num_y=0; num_y<NUMBER_OF_PIXEL_Y;num_y++){
          shower[layers][num_z][num_y][0] = TMath::Power(10,shower[layers][num_z][num_y][0]*EN_NORM) ;
          // verifico la lateral width sull'asse z (sarebbero le ascisse)
          if(pid == 0){
            x_photons[layers] += shower[layers][num_z][num_y][0]*(num_z-NUMBER_OF_PIXEL_Z/2)/1E6;
            x2_photons[layers] += shower[layers][num_z][num_y][0]*TMath::Power(num_z-NUMBER_OF_PIXEL_Z/2,2)/1E6;
          }
          if(pid == 1 || pid == -1){
            x_leptons[layers] += shower[layers][num_z][num_y][0]*(num_z-NUMBER_OF_PIXEL_Z/2)/1E6;
            x2_leptons[layers] += shower[layers][num_z][num_y][0]*TMath::Power(num_z-NUMBER_OF_PIXEL_Z/2,2)/1E6;
          }
          en_inside_layer += shower[layers][num_z][num_y][0]/1E6;
        }
      }
      if(pid == 0){
        hist_mean_lateral_layer[0]->SetBinContent(layers+1, en_in/1.E6,
        TMath::Sqrt(TMath::Abs(x2_photons[layers]/en_inside_layer - TMath::Power(x_photons[layers]/en_inside_layer,2))));
        // cout<<pid<<"\t"<<en_in/1E6<<"\t"<<layers<<"\t"<<hist_mean_lateral_layer[0]->GetBinContent(layers, en_in/1.E6)<<endl;
      }
      if(pid == 1 || pid == -1){
        hist_mean_lateral_layer[1]->SetBinContent(layers+1, en_in/1.E6,
        TMath::Sqrt(TMath::Abs(x2_leptons[layers]/en_inside_layer - TMath::Power(x_leptons[layers]/en_inside_layer,2))));
        // cout<<pid<<"\t"<<layers<<"\t"<<hist_mean_lateral_layer[1]->GetBinContent(layers, en_in/1.E6)<<endl;
      }

    }
    if(TMath::Abs(pid)==1){
      count_electron++;
      hist_1 = hist_mean_lateral_layer[1]->ProjectionY();
      hist_1->Scale(1./11.);// num of layers bins
      hist_mean_1 ->Add( hist_1 );
    }

    if(TMath::Abs(pid)==0){
      count_photon++;
      hist_0 = hist_mean_lateral_layer[0]->ProjectionY();
      hist_0->Scale(1./11.); // num of layers bins
      hist_mean_0 ->Add( hist_0 );
    }
  }
  hist_mean_1->Scale(1./count_electron);
  hist_mean_1->Draw("HIST");
  hist_mean_1->SetLineWidth(2);
  hist_mean_1->SetLineColor(kRed);

  hist_mean_0->Scale(1./count_photon);
  hist_mean_0->Draw("HIST same");
  hist_mean_0->SetLineWidth(2);
  hist_mean_0->SetLineColor(kBlue);


  gStyle->SetOptStat(kFALSE);
  gPad->SetGrid();
  TLegend *legend = new TLegend(0.1,0.7,0.38,0.9);
  legend -> AddEntry(hist_mean_0, "Photons", "l");
  legend -> AddEntry(hist_mean_1, "e^{#pm}", "l");
  legend -> Draw();

}
