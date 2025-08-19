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

    Mesh(string name_, int ni, int nj) : name(name_), NI(ni), NJ(nj) {}  
    Mesh(string name_,int ni,int nj,const string& filename_xx,const string& filename_yy):name(name_), NI(ni-1), NJ(nj-1){
        
        xx = load_binary_file(filename_xx,NI+1,NJ+1);
        yy = load_binary_file(filename_yy,NI+1,NJ+1);
        set_mesh();

    }



    



    void set_mesh();
    void set_uniform_points();
    int check_mesh();
    int check_interior_cells();
    int set_ramp_boundary_types();
    int set_square_boundary_types();
    int check_mesh_cellwise();
    void test_boundary_normals();
    int check_cell_points();
    vector<vector<double>>  load_binary_file(const std::string& filename,int NI,int NJ);











};
