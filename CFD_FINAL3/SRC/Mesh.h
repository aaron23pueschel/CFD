#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include "Cell.h"
#include <fstream>


using namespace std;
class Mesh{
    public:


    string name;
    int NI;
    int NJ;
    
    
    vector<Cell*> interior_cells;
    vector<Cell*> ghost_cells;
    vector<vector<double>> xx;
    vector<vector<double>> yy;

    
    Mesh(){}
    Mesh(string name_, int ni, int nj) : name(name_), NI(ni), NJ(nj) {}  
    Mesh(string name_,int ni,int nj,const string& filename_xx,const string& filename_yy):name(name_), NI(ni), NJ(nj){
        
        xx = load_binary_file(filename_xx,NI,NJ);
        yy = load_binary_file(filename_yy,NI,NJ);
        set_mesh();

    }

    Mesh(string name_,const string& filename_xx,const string& filename_yy):name(name_){
        xx = load_csv_file(filename_xx);
        yy = load_csv_file(filename_yy);
        NI = xx.size();
        NJ = ((NI > 0) ? xx[0].size() : 0);

        //  for(int i=0;i<=NI;i++)
        //      for(int j=0;j<=NJ;j++)
        //          cout << xx[i][j] <<" "<<yy[i][j]<<endl;
        
        // cout << NI <<", "<<NJ;
        set_mesh_cells();


    }



    


    vector<vector<double>> load_csv_file(const string& filename);
    void set_mesh();
    int set_MMS_boundary_types();
    void set_uniform_points();
    void set_mesh_cells();
    int check_mesh();
    void set_airfoil_boundary_types();
    int check_interior_cells();
    void print_mesh();
    void set_pointers();
    void check_divergence();
    int set_ramp_boundary_types();
    int set_square_boundary_types();
    int check_mesh_cellwise();
    void test_boundary_normals();
    int check_cell_points();
    void write_vector_to_binary(const std::vector<double>& data, const std::string& filename);
    void assemble_conserved();
    void print_conserved();
    
    vector<vector<double>>  load_binary_file(const std::string& filename,int NI,int NJ);











};
