#include "Cell.h"
#include "Mesh.h"
#include <cmath>
#include <array>
#include <fstream>
#include <sstream>
using namespace std;


void Mesh::set_uniform_points() {
    // allocate (NI+1) by (NJ+1)
    xx.assign(NI+1, vector<double>(NJ+1, 0.0));
    yy.assign(NI+1, vector<double>(NJ+1, 0.0));

    // uniform spacing
    double dx = 1.0 / NI;  // NI intervals → NI+1 points
    double dy = 1.0 / NJ;  // NJ intervals → NJ+1 points

    for (int i = 0; i <= NI; i++) {
        for (int j = 0; j <= NJ; j++) {
            xx[i][j] = j * dx;   // x along columns
            yy[i][j] = i * dy;   // y along rows
        }
    }
}




void Mesh::write_vector_to_binary(const vector<double>& data, const string& filename) {
    vector<vector<double>> array(NI, vector<double>(NJ));


    ofstream out("array.bin", ios::binary);
    for (const auto& row : array) {
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
    out.close();
}




void Mesh::print_mesh(){
    for (int i = 0; i<=NI; i++) {

        for (int j = 0; j <= NJ; j++) {
            cout << yy[i][j]<<"  ";

        }
        cout<<""<<endl;
    }



}

void Mesh::set_mesh_cells(){
    std::size_t rows = xx.size();
    std::size_t cols = (rows > 0 ? xx[0].size() : 0);

    std::cout << "Shape = (" << rows << ", " << cols << ")\n";


    for (int i = 0; i<NI; i++) {

            Cell* temp_cell = new Cell("Interior");
            temp_cell->set_points(
                xx[i][0],     yy[i][0], 
                xx[i][1],   yy[i][1],
                xx[i][2],   yy[i][2],
                xx[i][3],   yy[i][3]
            );


            temp_cell->initialize_areas();
            temp_cell->initialize_normals();
            temp_cell->i_idx = i;
            temp_cell->j_idx = i;
            interior_cells.push_back(temp_cell);
        
        }

    set_pointers();
    if(name=="Ramp")
        set_ramp_boundary_types();
    else if(name=="Airfoil")
        set_airfoil_boundary_types();
    else{
        set_square_boundary_types();
        cout << "Setting Square BC";
    }
        
}

void Mesh::set_pointers(){
    cout<<"setting pointers";
    for (Cell* c_from : interior_cells) {

        for (Cell* c_to : interior_cells){
            if(c_from != c_to){
                if (c_from->x11 == c_to->x12 && c_from->y11 == c_to->y12 &&
                    c_from->x21 == c_to->x22 && c_from->y21 == c_to->y22) {
                    c_from->cell_L = c_to;
                    c_to->cell_R   = c_from;
                }

                // Right neighbor: (x12,y12)-(x22,y22) matches (x11,y11)-(x21,y21)
                if (c_from->x12 == c_to->x11 && c_from->y12 == c_to->y11 &&
                    c_from->x22 == c_to->x21 && c_from->y22 == c_to->y21) {
                    c_from->cell_R = c_to;
                    c_to->cell_L   = c_from;
                }

                // Up neighbor: (x11,y11)-(x12,y12) matches (x21,y21)-(x22,y22)
                if (c_from->x11 == c_to->x21 && c_from->y11 == c_to->y21 &&
                    c_from->x12 == c_to->x22 && c_from->y12 == c_to->y22) {
                    c_from->cell_U = c_to;
                    c_to->cell_D   = c_from;
                }

                // Down neighbor: (x21,y21)-(x22,y22) matches (x11,y11)-(x12,y12)
                if (c_from->x21 == c_to->x11 && c_from->y21 == c_to->y11 &&
                    c_from->x22 == c_to->x12 && c_from->y22 == c_to->y12) {
                    c_from->cell_D = c_to;
                    c_to->cell_U   = c_from;
                }


            }
            else
                continue;


        }



        if (c_from->cell_L == nullptr) {
            Cell* ghost = new Cell("Ghost");
            c_from->cell_L = ghost;
            ghost->cell_R  = c_from;
            ghost_cells.push_back(ghost);
        }

        // Right ghost
        if (c_from->cell_R == nullptr) {
            Cell* ghost = new Cell("Ghost");
            c_from->cell_R = ghost;
            ghost->cell_L  = c_from;
            ghost_cells.push_back(ghost);
        }

        // Up ghost
        if (c_from->cell_U == nullptr) {
            Cell* ghost = new Cell("Ghost");
            c_from->cell_U = ghost;
            ghost->cell_D  = c_from;
            ghost_cells.push_back(ghost);
        }

        // Down ghost
        if (c_from->cell_D == nullptr) {
            Cell* ghost = new Cell("Ghost");
            c_from->cell_D = ghost;
            ghost->cell_U  = c_from;
            ghost_cells.push_back(ghost);
        }






    }




}







// void Mesh::set_mesh(){
//     Cell* temp_cells[NI][NJ];
//     for (int i = 0; i<NI; i++) {
//         for (int j = 0; j < NJ; j++) {
//             int i_idx_ =  i; //NI-i-1;
//             temp_cells[i_idx_][j] = new Cell("Interior");
//             temp_cells[i_idx_][j]->set_points(
//                 xx[i+1][j+1],     yy[i+1][j+1], 
//                 xx[i+1][j],   yy[i+1][j],
//                 xx[i][j+1],   yy[i][j+1],
//                 xx[i][j], yy[i][j]
//             );
//             temp_cells[i_idx_][j]->initialize_areas();
//             temp_cells[i_idx_][j]->initialize_normals();
//             temp_cells[i_idx_][j]->i_idx = i;
//             temp_cells[i_idx_][j]->j_idx = j;


//             interior_cells.push_back(temp_cells[i_idx_][j]);


//         }
//     }
    

//     auto attach_ghost = [&](Cell*& interior_slot, Cell* c, char dir_back) {
//         Cell* g = new Cell("Ghost");
//         g->cell_L = g->cell_R = g->cell_U = g->cell_D = nullptr;

//         // backlink from ghost to interior (opposite direction)
//         switch (dir_back) {
//         case 'L': {
//             // Interior points LEFT to ghost; ghost is on the left side.
//             interior_slot = g;      // c->cell_L = g
//             g->cell_R = c;          // ghost's right points back to interior

//             // Opposite normal on the shared interface:
//             // ghost's Right-face normal = - interior's Left-face normal
//             g->nx_R = -c->nx_L;  g->ny_R = -c->ny_L;
//             g->A_R  =  c->A_L;   // match edge length (optional)
//             break;
//         }
//         case 'R': {
//             interior_slot = g;      // c->cell_R = g
//             g->cell_L = c;

//             g->nx_L = -c->nx_R;  g->ny_L = -c->ny_R;
//             g->A_L  =  c->A_R;
//             break;
//         }
//         case 'D': {
//             interior_slot = g;      // c->cell_D = g
//             g->cell_U = c;

//             g->nx_U = -c->nx_D;  g->ny_U = -c->ny_D;
//             g->A_U  =  c->A_D;
//             break;
//         }
//         case 'U': {
//             interior_slot = g;      // c->cell_U = g
//             g->cell_D = c;

//             g->nx_D = -c->nx_U;  g->ny_D = -c->ny_U;
//             g->A_D  =  c->A_U;
//             break;
//         }
//         default:
//             // invalid dir: ignore or throw
//             break;
//     }

//         interior_slot = g;
//         ghost_cells.push_back(g);
//     };

//     for (int i = 0; i < NI; ++i) {
//         for (int j = 0; j < NJ; ++j) {
//             Cell* c = temp_cells[i][j];

//             // Down (i-1)
//             if (i > 0) {
//                 c->cell_D = temp_cells[i-1][j];
//                 temp_cells[i-1][j]->cell_U = c;   // reciprocal
//             } else {
//                 attach_ghost(c->cell_D, c, 'U');  // interior points Down -> ghost's Up points back
//             }

//             // Up (i+1)
//             if (i < NI - 1) {
//                 c->cell_U = temp_cells[i+1][j];
//                 temp_cells[i+1][j]->cell_D = c;   // reciprocal
//             } else {
//                 attach_ghost(c->cell_U, c, 'D');  // interior points Up -> ghost's Down points back
//             }

//             // Left (j-1)
//             if (j > 0) {
//                 c->cell_L = temp_cells[i][j-1];
//                 temp_cells[i][j-1]->cell_R = c;   // reciprocal
//             } else {
//                 attach_ghost(c->cell_L, c, 'R');  // interior points Left -> ghost's Right points back
//             }

//             // Right (j+1)
//             if (j < NJ - 1) {
//                 c->cell_R = temp_cells[i][j+1];
//                 temp_cells[i][j+1]->cell_L = c;   // reciprocal
//             } else {
//                 attach_ghost(c->cell_R, c, 'L');  // interior points Right -> ghost's Left points back
//             }
//         }
//     }

//     if(name=="Ramp")
//         set_ramp_boundary_types();
//     else
//         throw(runtime_error("Error setting boundary types"));

// }


int Mesh::check_cell_points() {
    double sum = 0.0;

    for (Cell* c : interior_cells) {
        if (c->cell_L && c->cell_L->name != "Ghost") {
            sum += abs(c->x11 - c->cell_L->x12);
            sum += abs(c->x21 - c->cell_L->x22);
        }

        if (c->cell_R && c->cell_R->name != "Ghost") {
            sum += abs(c->x12 - c->cell_R->x11);
            sum += abs(c->x22 - c->cell_R->x21);
        }

        if (c->cell_D && c->cell_D->name != "Ghost") {
            sum += abs(c->y11 - c->cell_D->y21);
            sum += abs(c->y12 - c->cell_D->y22);
        }

        if (c->cell_U && c->cell_U->name != "Ghost") {
            sum += abs(c->y21 - c->cell_U->y11);
            sum += abs(c->y22 - c->cell_U->y12);
        }
    }

    cout << "Total corner mismatch sum: " << sum << "\n";
    return 0;
}






















int Mesh::check_mesh_cellwise() {
    cout << "Checking cellwise normals";
    
    std::size_t rows = interior_cells.size();
    

    std::cout << "Shape = (" << rows << ", ";
    for (Cell* c : interior_cells) {
        cout << "Here";
        double sum_x = 0.0;
        double sum_y = 0.0;

        // Left face
        sum_x += c->nx_L + c->cell_L->nx_R;
        sum_y += c->ny_L + c->cell_L->ny_R;

        // Right face
        sum_x += c->nx_R + c->cell_R->nx_L;
        sum_y += c->ny_R + c->cell_R->ny_L;

        // Down face
        sum_x += c->nx_D + c->cell_D->nx_U;
        sum_y += c->ny_D + c->cell_D->ny_U;

        // Up face
        sum_x += c->nx_U + c->cell_U->nx_D;
        sum_y += c->ny_U + c->cell_U->ny_D;

        cout << "Cell " << "Hi"
                  << " face-pair normal sum = ("
                  << sum_x << ", " << sum_y << ")\n" << c->midpoint_x << ",  "<< c->midpoint_y;
    }

    return 0;
}

int Mesh::check_mesh(){
    double sum_x = 0.0;
    double sum_y = 0.0;

    for (Cell* cell : interior_cells) {
        sum_x += cell->nx_L + cell->nx_R + cell->nx_U + cell->nx_D;
        sum_y += cell->ny_L + cell->ny_R + cell->ny_U + cell->ny_D;
    }

    cout << "Sum of normals (x, y) = (" << sum_x << ", " << sum_y << ")\n";



return 0;


}

int Mesh::check_interior_cells(){

    bool all_neighbors_ok = true;

    for (Cell* c : interior_cells) {
        if (c->cell_L == nullptr) {
            cout << "Missing LEFT neighbor for cell " << c << "\n";
            all_neighbors_ok = false;
        }
        if (c->cell_R == nullptr) {
            cout << "Missing RIGHT neighbor for cell " << c << "\n";
            all_neighbors_ok = false;
        }
        if (c->cell_U == nullptr) {
            cout << "Missing UP neighbor for cell " << c << "\n";
            all_neighbors_ok = false;
        }
        if (c->cell_D == nullptr) {
            cout << "Missing DOWN neighbor for cell " << c << "\n";
            all_neighbors_ok = false;
        }
    }

    if (all_neighbors_ok) {
        cout << "✅ All interior cells have valid neighbors.\n";
    } else {
        cout << "⚠️ Some neighbors are missing.\n";
    }
    return 0;
}





void Mesh::print_conserved() {
    vector<vector<double>> matrix(NI, vector<double>(NJ, 0.0));
    
    for (Cell* c : interior_cells) {

        matrix[c->i_idx][c->j_idx] = c->U[1]/c->U[0];
    }

    // Write to binary file (row-major order)
    ofstream out("volume2.bin", ios::binary);
    for (int i = 0; i < NI; i++) {
        out.write(reinterpret_cast<char*>(matrix[i].data()), NJ * sizeof(double));
    }
    out.close();
}


void Mesh::check_divergence(){

    
    bool flag = false;
    for (Cell* c : interior_cells) {
        double temp_sum_x = c->nx_L*c->A_L +c->nx_R*c->A_R+c->nx_U*c->A_U +c->nx_D*c->A_D;
        double temp_sum_y = c->ny_L*c->A_L +c->ny_R*c->A_R+c->ny_U*c->A_U +c->ny_D*c->A_D;
        if(temp_sum_x > 1e-10){
            flag = true;
            cout <<"DIVERGENCE THEORY NOT MET IN X: "<<temp_sum_x;
            
        }
        if(temp_sum_x > 1e-10){
            flag = true;
            cout <<"DIVERGENCE THEORY NOT MET IN Y: "<<temp_sum_y;
            
        }
        if(flag)
            return;

    }


    cout <<"DIVERGENCE THEORY MET!!!!! ";


}










// void Mesh::print_conserved(){
//     vector<vector<double>> matrix(NI, vector<double>(NJ, 0.0));
//     for (Cell* c : interior_cells) {
        
//         matrix[c->i_idx][c->j_idx] = c->Residual[1];        
        
    
    
//     }

//     for(int i=0;i<NI;i++){
//         for(int j=0; j<NJ;j++){
//             cout <<matrix[i][j] << " ";
//         }
//        cout << endl; 
//     }




// }

void Mesh::set_airfoil_boundary_types(){
    cout << "Setting boundary types"<<endl;
    for (Cell* c : ghost_cells) {
       
        if(c->cell_D != nullptr){
            c->type = 2; 
        }   
        if(c->cell_R !=nullptr)
            c->type = 0; // outflow
        if(c->cell_L !=nullptr)
            c->type = 0; //slip
        if(c->cell_U != nullptr){
            Cell* cell_temp = c->cell_U;
            for (Cell* c_ : ghost_cells) {

                if(c_->cell_U && c_->cell_U!=cell_temp && cell_temp->x22==c_->cell_U->x21 &&cell_temp->x21==c_->cell_U->x22
                                 && cell_temp->y22==c_->cell_U->y21 &&cell_temp->y21==c_->cell_U->y22){

                    c->cell_U->cell_D = c_->cell_U;
                    //c_->cell_U = c->cell_U->cell_D;
                    cout << "Found matching condition"<<endl;

                    break;
                }

            }
            c->type =1;

        }   


    }



}


int Mesh::set_ramp_boundary_types(){




    for (Cell* c : ghost_cells) {
       
        if(c->cell_D != nullptr){
            if(c->cell_D->y11 <.6){
                c->type = 2; // inflow
            }
            else
                c->type = 1; // slip
        }   
        if(c->cell_L !=nullptr)
            c->type = 0; 
        if(c->cell_R !=nullptr)
            c->type = 1; //slip
        if(c->cell_U != nullptr)
            c->type = 1;

    }

    return 0;



}



int Mesh::set_square_boundary_types(){
    cout << "Setting square inflow";
    for (Cell* c : ghost_cells) {

        if(c->cell_D != nullptr)
            c->type = 1; // slip
        if(c->cell_L !=nullptr)
            c->type = 0; // outflow
        if(c->cell_R !=nullptr){
            c->type = 2; //inflow

        }
        if(c->cell_U != nullptr)
            c->type = 1;

        

    }

    return 0;

}


vector<vector<double>> Mesh::load_csv_file(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> data;
    string line;

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));  // convert to double
        }

        data.push_back(row);
    }
    
    return data;
}

vector<vector<double>> Mesh::load_binary_file(const string& filename, int ni, int nj) {
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file " + filename);
    }

    vector<double> buffer((ni+1) * (nj+1));
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(double));

    // reshape into 2D
    vector<vector<double>> result(ni+1, vector<double>(nj+1));
    for (int i = 0; i <= ni; ++i) {
        for (int j = 0; j <= nj; ++j) {
            result[i][j] = buffer[i * (nj+1) + j];
        }
    }
    return result;
}

void Mesh::test_boundary_normals() {
    for (Cell* c : ghost_cells) {
        cout << "Ghost cell:\n";

        if (c->cell_L) {
            cout << "  Left  normal = (" << c->nx_L << ", " << c->ny_L << ")\n";
        }
        if (c->cell_R) {
            cout << "  Right normal = (" << c->nx_R << ", " << c->ny_R << ")\n";
        }
        if (c->cell_U) {
            cout << "  Up    normal = (" << c->nx_U << ", " << c->ny_U << ")\n";
        }
        if (c->cell_D) {
            cout << "  Down  normal = (" << c->nx_D << ", " << c->ny_D << ")\n";
        }
    }
}