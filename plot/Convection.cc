/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Pankaj Kumar, MSc 2017.
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>

#include <deal.II/base/logstream.h>



using namespace dealii;

template <int dim>
class Convection
{
public:
  Convection ();
  ~Convection();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void assemble_system_2 ();
  void solve ();
  void refine_grid (const unsigned int min_grid_level, const unsigned int max_grid_level);
  void output_results () const;

  double solution_bdf(
    const double& sol_val,
    const double& rhs_val,
    const double& v_val,
    const Tensor<1, dim>& sol_grad,
    const Tensor<1, dim>& v_grad,
    const Tensor<1, dim>& beta
  ) const;

  double solution_bdf1(
    const double& sol_old,
    const double& sol_old_old
  ) const;

  double advection_cell_operator(
    const double& u_val,
    const double& v_val,
    const Tensor<1, dim>& u_grad,
    const Tensor<1, dim>& v_grad,
    const Tensor<1, dim>& beta
  ) const;

  double advection_face_operator(
    const double& u_val,
    const double& v_val,
    const Tensor<1, dim>& beta,
    const Tensor<1, dim>& normal
  ) const;

  double lhs_operator(
    const double& u_val,
    const double& v_val,
    const Tensor<1, dim>& u_grad,
    const Tensor<1, dim>& v_grad,
    const double& alpha,
    const Tensor<1, dim>& beta
  ) const;

  double streamline_diffusion(
    const Tensor<1, dim>& u_grad,
    const Tensor<1, dim>& v_grad,
    const Tensor<1, dim>& beta
  ) const;

  Triangulation<dim>   triangulation;

  FE_Q<dim>            fe;
  FE_Q<dim>            fe_velocity;

  DoFHandler<dim>      dof_handler;
  DoFHandler<dim>      dof_vel_handler;

  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       old_solution;
  Vector<double>       old_old_solution;
  Vector<double>       solution;
  Vector<double>       system_rhs;

  unsigned int         timestep_number;
  double               time_step;
  double               time;

  double               theta_imex;
  double               theta_skew;

  const double         nu = 1.0;//0.001;
};

template <int dim>
class TemperatureInitialValues : public Function<dim>
{
public:
TemperatureInitialValues () : Function<dim>(1) {}
virtual double value (const Point<dim>   &p,
					  const unsigned int  component = 0) const;
};

template <int dim>
double
TemperatureInitialValues<dim>::value (const Point<dim> &p,
        							  const unsigned int component) const
{
	double PI = 3.14159265358979323846;


	double initial_exp = (1 - p[0]*p[0])*(1 - p[1]*p[1]);//0.0;//(std::sin(PI*p[0]))*(std::sin(PI*p[1]));

	return initial_exp;
}


template <int dim>
class TemperatureExactSol : public Function<dim>
{
public:
TemperatureExactSol (const double& time) : Function<dim>(1), time(time) {}
virtual double value (const Point<dim>   &p,
					  const unsigned int  component = 0) const;
const double time;
};

template <int dim>
double
TemperatureExactSol<dim>::value (const Point<dim> &p,
        							  const unsigned int component) const
{

	double initial_exp = std::exp(-time)*(1 - p[0]*p[0])*(1 - p[1]*p[1]);

	return initial_exp;
}

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide ()
    :
    Function<dim>(),
    period (0.2)
  {}
  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
private:
  const double period;
};


template<int dim>
class RightHandSide1 : public Function<dim>
{
public:
  RightHandSide1 (const double& time) : Function<dim>(), time(time) {}
  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
  const double time;
//private:
//  static const Point<dim> center_point;
};
/*
template <>
const Point<2> RightHandSide1<2>::center_point = Point<2> (-0.75, -0.75);
*/
template<int dim>
double RightHandSide1<dim> ::value(const Point<dim> &p,
		                    const unsigned int component) const{
	 /*
    double PI = 3.14159265358979323846;

    double right_exp = std::exp(-time)*std::sin(PI*p[0])*( -std::sin(PI*p[1])
                                                           - 2*0.001*PI*PI*std::sin(PI*p[1])
                                                           + PI*std::cos(PI*p[0]));

    return right_exp;


    Assert (component == 0, ExcIndexRange (component, 0, 1));
    const double diameter = 0.1;
    return ( (p-center_point).norm_square() < diameter*diameter ?
             .1/std::pow(diameter,dim) :
             0);*/

	return std::exp(-time)*(5 - 3*p[0]*p[0] -3*p[1]*p[1] + p[0]*p[0]*p[1]*p[1]);
}

template<int dim>
class VelocityU : public Function<dim>
{
public:
	VelocityU() : Function<dim>(){}
    virtual double value (const dealii::Point<dim>   &p,
                          const unsigned int  component = 0) const;

};

template <int dim>
double
VelocityU<dim>::value (const Point<dim>  &p,
							  const unsigned int component) const
{

	double PI = 3.14159265358979323846;


	return (2*p[0]*p[0] - std::pow(p[0],4) - 1)*(p[1] - p[1]*p[1]*p[1]);//2.0;//(std::sin(PI*p[0]))*(std::sin(PI*p[1]));

}

template<int dim>
class VelocityV : public Function<dim>
{
public:
	VelocityV() : Function<dim>(){}
    virtual double value (const dealii::Point<dim>   &p,
                          const unsigned int  component = 0) const;

};

template <int dim>
double
VelocityV<dim>::value (const Point<dim>  &p,
							  const unsigned int component) const
{

	double PI = 3.14159265358979323846;


	return -(2*p[1]*p[1] - std::pow(p[1],4) - 1)*(p[0] - p[0]*p[0]*p[0]);//1+0.8*std::sin(8*PI*p[0]);//(std::cos(PI*p[0]))*(std::cos(PI*p[1]));

}

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};




template<int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int component) const
{
  Assert (component == 0, ExcInternalError());
  Assert (dim == 2, ExcNotImplemented());

  const double time = this->get_time();
  const double point_within_period = (time/period - std::floor(time/period));

  if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
    {
      if ((p[0] > 0.5) && (p[1] > -0.5))
        return 0.1;
      else
        return 0;
    }
  else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
    {
      if ((p[0] > -0.5) && (p[1] > 0.5))
        return 1;
      else
        return 0;
    }
  else
    return 0;
}


template<int dim>
double BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                                   const unsigned int component) const
{   /*
	  double value = 1.0;

	  for (unsigned int i=0; i<dim; ++i)
	    value *= std::sin(k_num*numbers::PI*p[i]);

	  return value;
	  */
    Assert(component == 0, ExcInternalError());
    return 0;
}

template<int dim>
double Convection<dim>::solution_bdf1(
  const double& sol_old,
  const double& sol_old_old
) const{
  return(
	+ (2.0)*sol_old - (0.5)*sol_old_old
	);
}

template<int dim>
double Convection<dim>::solution_bdf(
  const double& sol_val,
  const double& rhs_val,
  const double& v_val,
  const Tensor<1, dim>& sol_grad,
  const Tensor<1, dim>& v_grad,
  const Tensor<1, dim>& beta
) const{
  return(
	+ sol_val*v_val
	- time_step*contract(beta , sol_grad )* v_val
//    + time_step * nu * sol_grad * v_grad
    + time_step * rhs_val*v_val
  );
}

template<int dim>
double Convection<dim>::advection_cell_operator(
  const double& u_val,
  const double& v_val,
  const Tensor<1, dim>& u_grad,
  const Tensor<1, dim>& v_grad,
  const Tensor<1, dim>& beta
) const {
  return (
    + (1 - theta_skew) * contract(beta, u_grad) * v_val
    + (0 - theta_skew) * contract(beta, v_grad) * u_val
    );
}


template<int dim>
double Convection<dim>::advection_face_operator(
  const double& u_val,
  const double& v_val,
  const Tensor<1, dim>& beta,
  const Tensor<1, dim>& normal
) const {
  return (
    + theta_skew * contract(beta, normal) * v_val * u_val
    );
}

template<int dim>
double Convection<dim>::lhs_operator(
  const double& u_val,
  const double& v_val,
  const Tensor<1, dim>& u_grad,
  const Tensor<1, dim>& v_grad,
  const double& alpha,
  const Tensor<1, dim>& beta
) const {
  return (
    + theta_imex * advection_cell_operator(u_val, v_val, u_grad, v_grad, beta)
    + theta_skew * alpha * (u_grad * v_grad)
    + alpha * (u_grad * v_grad)
    );
}

template<int dim>
double Convection<dim>::streamline_diffusion(
  const Tensor<1, dim>& u_grad,
  const Tensor<1, dim>& v_grad,
  const Tensor<1, dim>& beta
) const {
  const double dt = time_step;
  return dt*dt/6*contract(beta, u_grad) * contract(beta, v_grad);
}

template <int dim>
Convection<dim>::Convection ()
  :
  fe (1),
  fe_velocity(1),
  dof_handler (triangulation),
  dof_vel_handler(triangulation),
  timestep_number(0),
  time_step(1. / 500),
  time(0),
  theta_imex(0.5),
  theta_skew(0.5)
{}

template <int dim>
Convection<dim>::~Convection (){
  dof_handler.clear ();
}

template <int dim>
void Convection<dim>::make_grid ()
{
//  GridGenerator::hyper_L(triangulation);
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (4);

  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void Convection<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  dof_vel_handler.distribute_dofs(fe_velocity);

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);

  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            constraints);


  constraints.close();

  DynamicSparsityPattern  c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, c_sparsity, constraints, /*keep_constrained_dofs = */ true);

  sparsity_pattern.copy_from(c_sparsity);

  system_matrix.reinit (sparsity_pattern);

  old_solution.reinit(dof_handler.n_dofs());
  old_old_solution.reinit(dof_handler.n_dofs());
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}



template <int dim>
void Convection<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);

  system_matrix = 0;
  system_rhs    = 0;

  RightHandSide1<dim> right_hand_side(time);
  VelocityU<dim>       velocity_U;
  VelocityV<dim>       velocity_V;

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);


  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double >                 old_values (n_q_points);
  std::vector<double >                 old_old_values (n_q_points);

  std::vector<Tensor<1,dim> >          old_grad (n_q_points);
  std::vector<Tensor<1,dim> >          old_old_grad (n_q_points);
  std::vector<double>                  rhs_values_t (n_q_points);
  std::vector<double>                  rhs_values_t_1 (n_q_points);

  std::vector<double>                  velocity_U_values (n_q_points);
  std::vector<double>                  velocity_V_values (n_q_points);


  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      const double delta = 0.1 * cell->diameter ();
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.get_function_values (old_solution, old_values);
      fe_values.get_function_gradients(old_solution, old_grad);
      fe_values.get_function_values (old_old_solution, old_old_values);
      fe_values.get_function_gradients (old_old_solution, old_old_grad);

//      right_hand_side.set_time(time);
      right_hand_side.value_list (fe_values.get_quadrature_points(),
                                  rhs_values_t);
/*      right_hand_side.set_time(time-time_step);
      right_hand_side.value_list (fe_values.get_quadrature_points(),
                                  rhs_values_t_1);
*/
      velocity_U.value_list(fe_values.get_quadrature_points(), velocity_U_values);
      velocity_V.value_list(fe_values.get_quadrature_points(), velocity_V_values);


      for (unsigned int q_index=0; q_index<n_q_points; ++q_index){

         Tensor<1, dim> velocity_values;
		 velocity_values[0] = velocity_U_values[q_index];
		 velocity_values[1] = velocity_V_values[q_index];

		 const double& exp_sol_val = 2.0*old_values[q_index] - old_old_values[q_index];
		 const Tensor<1, dim>& exp_sol_grad = 2.0*old_grad[q_index] - old_old_grad[q_index];

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {

        	const double& v_val          =  fe_values.shape_value(i, q_index);
    	    const Tensor<1, dim>& v_grad =  fe_values.shape_grad(i, q_index);

            for (unsigned int j=0; j<dofs_per_cell; ++j){

            	const double& u_val =  fe_values.shape_value(j, q_index);
        	    const Tensor<1, dim>& u_grad =  fe_values.shape_grad(j, q_index);


				cell_matrix(i, j) += (
				  + u_val * v_val
				  + time_step * this->lhs_operator(u_val, v_val, u_grad, v_grad, nu, velocity_values)
				  + streamline_diffusion(u_grad, v_grad, velocity_values)
				  ) * fe_values.JxW(q_index);
            }


		        cell_rhs(i) += (
//		          + solution_bdf(old_values[q_index], rhs_values_t[q_index], v_val, old_grad[q_index], v_grad, velocity_values)
		          + old_values[q_index]*v_val
		          + time_step * (
		              + rhs_values_t[q_index]  * v_val
		              - (1 - theta_imex) * advection_cell_operator( exp_sol_val, v_val, exp_sol_grad, v_grad, velocity_values)
		              - (1 - theta_imex) * nu * (exp_sol_grad * v_grad)
		          )
		          ) * fe_values.JxW(q_index);
          }
      }
      cell->get_dof_indices (local_dof_indices);


      constraints.distribute_local_to_global(cell_matrix,
                                          cell_rhs,
                                          local_dof_indices,
                                          system_matrix,
                                          system_rhs);

    }


  BoundaryValues<dim> boundary_values_function;
  boundary_values_function.set_time(time);

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            boundary_values_function,
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

}

template <int dim>
void Convection<dim>::assemble_system_2 ()
{
  QGauss<dim>  quadrature_formula(2);

  const RightHandSide1<dim> right_hand_side(time);
  VelocityU<dim>       velocity_U;
  VelocityV<dim>       velocity_V;

  system_matrix = 0;
  system_rhs    = 0;

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);


  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double >                 old_values (n_q_points);
  std::vector<Tensor<1,dim> >          old_grad (n_q_points);
  std::vector<double>                  rhs_values_t (n_q_points);
  std::vector<double>                  rhs_values_t_1 (n_q_points);

  std::vector<double>                  velocity_U_values (n_q_points);
  std::vector<double>                  velocity_V_values (n_q_points);
  std::vector<Tensor<1,dim> >          velocity_field (n_q_points);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.get_function_values (old_solution, old_values);
      fe_values.get_function_gradients(old_solution, old_grad);


      right_hand_side.value_list (fe_values.get_quadrature_points(),
                                  rhs_values_t);


      velocity_U.value_list(fe_values.get_quadrature_points(), velocity_U_values);
      velocity_V.value_list(fe_values.get_quadrature_points(), velocity_V_values);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index){

    	 Tensor<1, dim> velocity_values;
		 velocity_values[0] = velocity_U_values[q_index];
		 velocity_values[1] = velocity_V_values[q_index];

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += ((fe_values.shape_value(j, q_index) *
            		               fe_values.shape_value(i, q_index)
            		               +
            		               time_step*0.5*(contract(velocity_values , fe_values.shape_grad (j, q_index) )
				                                * fe_values.shape_value (i, q_index)
					                            - contract(velocity_values , fe_values.shape_grad (i, q_index) )
				                                * fe_values.shape_value (j, q_index) )
            		               +
            		               nu*time_step *
            		               fe_values.shape_grad (j, q_index) *
                                       fe_values.shape_grad (i, q_index) )*
                                       fe_values.JxW (q_index));

            cell_rhs(i) += ((old_values[q_index]*fe_values.shape_value (i, q_index)
            		      +  time_step * rhs_values_t[q_index] * fe_values.shape_value (i, q_index)
                               )*
                            fe_values.JxW (q_index));
          }
      }
      cell->get_dof_indices (local_dof_indices);
     constraints.distribute_local_to_global(cell_matrix,
                                          cell_rhs,
                                          local_dof_indices,
                                          system_matrix,
                                          system_rhs);

    }


  BoundaryValues<dim> boundary_values_function;
  boundary_values_function.set_time(time);

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            boundary_values_function,
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

}


template <int dim>
void Convection<dim>::solve ()
{
/*  SolverControl           solver_control (1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<>              solver (solver_control);

  PreconditionSSOR<> preconditioner;//PreconditionIdentity()
  preconditioner.initialize(system_matrix, 1.0);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

*/
	int    vel_max_its     = 5000;
	double vel_eps         = 1e-9;
	int    vel_Krylov_size = 30;

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

	SolverControl solver_control (vel_max_its, vel_eps*system_rhs.l2_norm());
	{
		SolverGMRES<Vector<double>> gmres1 (solver_control,
						   SolverGMRES<>::AdditionalData (vel_Krylov_size));
		gmres1.solve (system_matrix, solution, system_rhs, preconditioner);
	}



  std::cout << "   " << solver_control.last_step()
            << " GMRES iterations needed to obtain convergence."
            << std::endl;


  constraints.distribute (solution);
}

template <int dim>
void Convection<dim>::refine_grid(const unsigned int min_grid_level,
		                     const unsigned int max_grid_level){

	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

	KellyErrorEstimator<dim>::estimate(dof_handler,
			                            QGauss<dim-1>(fe.degree+2),
			                            typename FunctionMap<dim>::type(),
			                            solution,
			                            estimated_error_per_cell);

	GridRefinement::refine_and_coarsen_fixed_number(triangulation,
			                                         estimated_error_per_cell,
			                                         0.6, 0.4);

	if(triangulation.n_levels() > max_grid_level){
		for(typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active(max_grid_level);
				cell != triangulation.end(); ++cell){
			cell->clear_refine_flag();
		}
	}
	for(typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active(min_grid_level);
			cell != triangulation.end_active(min_grid_level); ++cell)
		cell->clear_coarsen_flag();

	SolutionTransfer<dim> solution_transfer(dof_handler);

	Vector<double> previous_solution;
	previous_solution = solution;


	triangulation.prepare_coarsening_and_refinement();
	solution_transfer.prepare_for_coarsening_and_refinement(previous_solution);


	triangulation.execute_coarsening_and_refinement();
	setup_system();

	solution_transfer.interpolate(previous_solution, solution);

	constraints.distribute(solution);

}



template <int dim>
void Convection<dim>::output_results () const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "temperature");
    data_out.build_patches();
    const std::string filename = "solution-"
                                 + Utilities::int_to_string(timestep_number, 3) +
                                 ".vtk";
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}




template <int dim>
void Convection<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  std::ofstream error_out;
  error_out.open("l2_error.dat");

  make_grid();
  setup_system ();

  unsigned int pre_refinement_step = 0;
  const unsigned int n_adaptive_pre_refinement_steps = 5;
  const unsigned int initial_global_refinement = 3;
  Vector<float> difference_per_cell (triangulation.n_active_cells());
  double L2_error ;

start_time_iteration:

  timestep_number = 0;
  time            = 0;


  VectorTools::interpolate(dof_handler,
		                   TemperatureInitialValues<dim>(),//ZeroFunction<dim>(),
                           old_solution);

  /*
  VectorTools::project (dof_handler,
                        constraints,
                        QGauss<dim>(2),
                        TemperatureInitialValues<dim>(),
                        solution);
*/
  solution = old_solution;
  output_results();
//  old_solution = 0;
//  solution     = 0;

   do{

      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      assemble_system ();

      solve ();
      output_results ();

      if((timestep_number ==0)&& (pre_refinement_step < n_adaptive_pre_refinement_steps)){

    	  refine_grid(initial_global_refinement,
    			  initial_global_refinement + n_adaptive_pre_refinement_steps);
    	  ++pre_refinement_step;
    	  old_solution.reinit(solution.size());
    	  system_rhs.reinit(solution.size());

    	  goto start_time_iteration;

      }
      else if ((timestep_number > 0) && (timestep_number % 5 == 0)){

    	  refine_grid(initial_global_refinement,
    			  initial_global_refinement + n_adaptive_pre_refinement_steps);
     	  old_solution.reinit(solution.size());
          system_rhs.reinit(solution.size());
      }
      time += time_step;
      ++timestep_number;


      VectorTools::integrate_difference (dof_handler,
                                         solution,
                                         TemperatureExactSol<dim>(time),
                                         difference_per_cell,
                                         QGauss<dim>(3),
                                         VectorTools::L2_norm);
      L2_error = difference_per_cell.l2_norm();
      error_out << time <<"  "<<L2_error << std::endl;

      old_old_solution = old_solution;
      old_solution = solution;
      solution = 0;
  }while (time <= 1.0);


}



int main ()
{

  try
    {
      using namespace dealii;
      deallog.depth_console(0);

      Convection<2> heat_equation_solver;
      heat_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
