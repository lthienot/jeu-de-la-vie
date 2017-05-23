#include "omp.h"
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>
static unsigned couleur = 0xFFFF00FF; // Yellow

unsigned version = 0;

void first_touch_v3 (void);

void init_v2(void);

unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);
unsigned compute_v4 (unsigned nb_iter);
unsigned compute_v5 (unsigned nb_iter);
unsigned compute_v6 (unsigned nb_iter);
unsigned compute_v7 (unsigned nb_iter);
unsigned compute_v8 (unsigned nb_iter);
  
void_func_t first_touch [] = {
  NULL,
  NULL,
  NULL,
  first_touch_v3,
  first_touch_v3,
  first_touch_v3,
  first_touch_v3,
  first_touch_v3,
  NULL,
};

void_func_t init [] = {
  NULL,
  NULL,
  init_v2,
  NULL,
  NULL,
  init_v2,
  NULL,
  init_v2,
  NULL,
};


int_func_t compute [] = {
  compute_v0,
  compute_v1,
  compute_v2,
  compute_v3,
  compute_v4,
  compute_v5,
  compute_v6,
  compute_v7,
  compute_v8,
};

char *version_name [] = {
  "Séquentielle simple",
  "Séquentielle tuilée",
  "Séquentielle tuilée optimisée",
  "OpenMP (for)",
  "OpenMP tuilée (for)",
  "OpenMP optimisée (for)",
  "OpenMP tuilée (task)",
  "OpenMP optimisée (task)",
  "OpenCL",
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1
};

int count_neighbours(int i, int j)
{
  int count = 0;
  if(i>0 && j>0 && cur_img(i-1,j-1))
    count++;
  if(j>0 && cur_img(i,j-1))
    count++;
  if(i<DIM-1 && j>0 && cur_img(i+1,j-1))
    count++;
  if(i>0 && cur_img(i-1,j))
    count++;
  if(i<DIM-1 && cur_img(i+1,j) )
    count++;
  if(i>0 && j<DIM-1 && cur_img(i-1,j+1))
    count++;
  if(j<DIM-1 && cur_img(i,j+1))
    count++;
  if(i<DIM-1 && j<DIM-1 && cur_img(i+1,j+1) )
    count++;
  return count;

}
///////////////////////////// Version séquentielle simple


unsigned compute_v0 (unsigned nb_iter)
{

  /* for (unsigned it = 1; it <= nb_iter; it ++) { */
  /*   for (int i = 0; i < DIM; i++) */
  /*     for (int j = 0; j < DIM; j++) */
  /* 	next_img (i, j) = cur_img (j, i); */
    
  /*   swap_images (); */
  /* } */
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
      for (int i = 0; i < DIM; i++)
	for (int j = 0; j < DIM; j++)
	  {
	    int count = count_neighbours(i,j);
	    int current_img = cur_img(i,j);
	    if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
	      next_img(i,j) = 0;
	    else
	      next_img(i,j) = couleur;
	    if (current_img != next_img(i,j))
	      stop_it = 0;
	  }
      swap_images();

      if (stop_it)
	return it;
    }
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}

///////////////////////////// Version séquentielle tuilée

#define TILE_SIZE 32
#define TILE_NUMBER DIM/TILE_SIZE

unsigned compute_v1 (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
      for (int i = 0; i < TILE_NUMBER; i++)
	for (int j = 0; j < TILE_NUMBER; j++)
	  for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
	    for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
	      {
		int count = count_neighbours(iloc,jloc);
		int current_img = cur_img(iloc,jloc);
		if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
		  next_img(iloc,jloc) = 0;
		else
		  next_img(iloc,jloc) = couleur;
		if (current_img != next_img(iloc,jloc))
		  stop_it = 0;
	      }
      swap_images();
      if (stop_it)
	return it;
    }
  
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}

///////////////////////////// Version séquentielle tuilée optimisée

int** next;
int** courant;

void init_v2()
{
  next = malloc(TILE_NUMBER*sizeof(int*));
  courant = malloc(TILE_NUMBER*sizeof(int*));
  for (int y = 0; y < TILE_NUMBER; y++)
    {
      next[y] = malloc(TILE_NUMBER*sizeof(int));
      courant[y] = malloc(TILE_NUMBER*sizeof(int));
      for (int z = 0; z < TILE_NUMBER; z++)
	{
	  courant[y][z] = 0;
	  next[y][z] = 0;
	}
    }
}

unsigned compute_v2 (unsigned nb_iter) //ça marche pas !!!!
{
  int stop_it;
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      stop_it = 1;
      
      for (int i = 0; i < TILE_NUMBER; i++)
	for (int j = 0; j < TILE_NUMBER; j++)
	  {
	    if (courant[i][j] == 0 ||
		(courant[i][j] == 1 &&
		 ((i>0 && courant[i-1][j] == 0) ||
		  (i>0 && j>0 && courant[i-1][j-1] == 0) ||
		  (i>0 && j<TILE_NUMBER-1 && courant[i-1][j+1] == 0) ||
		  (i<TILE_NUMBER-1 && courant[i+1][j] == 0) ||
		  (i<TILE_NUMBER-1 && j>0 && courant[i+1][j-1] == 0) ||
		  (i<TILE_NUMBER-1 && j<TILE_NUMBER-1 && courant[i+1][j+1] == 0) ||
		  (j>0 && courant[i][j-1] == 0) ||
		  (j<TILE_NUMBER-1 && courant[i][j+1] == 0))))
	      {

		int stop_tuile = 1;
		for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
		  for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
		    {
		      int count = count_neighbours(iloc,jloc);
		      int current_img = cur_img(iloc,jloc);
		      if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
			next_img(iloc,jloc) = 0;
		      else
			next_img(iloc,jloc) = couleur;
		      if (current_img != next_img(iloc,jloc))
			{
			  stop_it = 0;
			  stop_tuile = 0;
			}
		    }
		next[i][j] = stop_tuile;
	      
	      }
	  }
      swap_images();
  
      int** tmp;
      tmp = courant;
      courant = next;
      next = tmp; 
       
      if (stop_it)
	return it;
    }
  
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}


///////////////////////////// Version OpenMP de base for

void first_touch_v3 ()
{
  int i,j ;

#pragma omp parallel for
  for(i=0; i<DIM ; i++) {
    for(j=0; j < DIM ; j += 512)
      next_img (i, j) = cur_img (i, j) = 0 ;
  }
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3(unsigned nb_iter)
{
  int ret = 0;

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
#pragma omp parallel for collapse(2)
      for (int i = 0; i < DIM; i++)
	for (int j = 0; j < DIM; j++)
	  {
	    int count = count_neighbours(i,j);
	    int current_img = cur_img(i,j);
	    if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
	      next_img(i,j) = 0;
	    else
	      next_img(i,j) = couleur;
	    if (current_img != next_img(i,j))
	      stop_it = 0;
	  }
      swap_images();

      if (stop_it)
	{
	  ret = it;
	  it = nb_iter+1;
	}
    }
  return ret;
}

///////////////////////////// Version OpenMP tuilée for


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v4(unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
#pragma omp parallel for collapse(2)
      for (int i = 0; i < TILE_NUMBER; i++)
	for (int j = 0; j < TILE_NUMBER; j++)
	  for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
	    for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
	      {
		int count = count_neighbours(iloc,jloc);
		int current_img = cur_img(iloc,jloc);
		if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
		  next_img(iloc,jloc) = 0;
		else
		  next_img(iloc,jloc) = couleur;
		if (current_img != next_img(iloc,jloc))
	  	  stop_it = 0;
	      }
      swap_images();
      if (stop_it)
	return it;
    }
  
  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenMP optimisée for


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v5(unsigned nb_iter)
{
  int stop_it;
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      stop_it = 1;
#pragma omp parallel for collapse(2)
      for (int i = 0; i < TILE_NUMBER; i++)
	for (int j = 0; j < TILE_NUMBER; j++)
	  {
	    if (courant[i][j] == 0 ||
		(courant[i][j] == 1 &&
		 ((i>0 && courant[i-1][j] == 0) ||
		  (i>0 && j>0 && courant[i-1][j-1] == 0) ||
		  (i>0 && j<TILE_NUMBER-1 && courant[i-1][j+1] == 0) ||
		  (i<TILE_NUMBER-1 && courant[i+1][j] == 0) ||
		  (i<TILE_NUMBER-1 && j>0 && courant[i+1][j-1] == 0) ||
		  (i<TILE_NUMBER-1 && j<TILE_NUMBER-1 && courant[i+1][j+1] == 0) ||
		  (j>0 && courant[i][j-1] == 0) ||
		  (j<TILE_NUMBER-1 && courant[i][j+1] == 0))))
	      {
		int stop_tuile = 1;
	      	for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
	      	  for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
	      	    {
		      int count = count_neighbours(iloc,jloc);
		      int current_img = cur_img(iloc,jloc);
		      if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
			next_img(iloc,jloc) = 0;
		      else
			next_img(iloc,jloc) = couleur;
		      if (current_img != next_img(iloc,jloc))
	      		{
	      		  stop_it = 0;
	      		  stop_tuile = 0;
	      		}
	      	    }
	      	next[i][j] = stop_tuile;
	      
	      }
	  }
      swap_images();
  
      int** tmp;
      tmp = courant;
      courant = next;
      next = tmp; 
      
      
      if (stop_it)
	return it;
    }
  
  return 0; // on ne s'arrête jamais
}

///////////////////////////// Version OpenMP tuilée task



// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v6(unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
#pragma omp parallel shared(stop_it)
      {
#pragma omp single
	for (int i = 0; i < DIM; i+=TILE_SIZE)
	  for (int j = 0; j < DIM; j+=TILE_SIZE)
#pragma omp task firstprivate(i,j)
	    {
	      for(int iloc = i; iloc < i+TILE_SIZE && iloc < DIM; iloc++)
		for(int jloc = j; jloc < j+TILE_SIZE && jloc < DIM; jloc++)
		  {
		    int count;
		    count = count_neighbours(iloc,jloc);
		    int current_img = cur_img(iloc,jloc);
		    if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
		      next_img(iloc,jloc) = 0;
		    else
		      next_img(iloc,jloc) = couleur;
		
		    if (current_img!=next_img(iloc,jloc))
		      stop_it = 0;
		  }
	    }
      }

      if (stop_it)
	return it;
      swap_images();
    }
  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenMP optimisée task


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v7(unsigned nb_iter)
{
  int stop_it;
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      stop_it = 1;
#pragma omp parallel
#pragma omp single
      {
	for (int i = 0; i < TILE_NUMBER; i++)
	  for (int j = 0; j < TILE_NUMBER; j++)
	    {
#pragma omp task firstprivate(i,j)
	      if (courant[i][j] == 0 ||
		  (courant[i][j] == 1 &&
		   ((i>0 && courant[i-1][j] == 0) ||
		    (i>0 && j>0 && courant[i-1][j-1] == 0) ||
		    (i>0 && j<TILE_NUMBER-1 && courant[i-1][j+1] == 0) ||
		    (i<TILE_NUMBER-1 && courant[i+1][j] == 0) ||
		    (i<TILE_NUMBER-1 && j>0 && courant[i+1][j-1] == 0) ||
		    (i<TILE_NUMBER-1 && j<TILE_NUMBER-1 && courant[i+1][j+1] == 0) ||
		    (j>0 && courant[i][j-1] == 0) ||
		    (j<TILE_NUMBER-1 && courant[i][j+1] == 0))))
		{
		  int stop_tuile = 1;
		  for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
		    for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
		      {
			int count = count_neighbours(iloc,jloc);
			int current_img = cur_img(iloc,jloc);
			if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
			  next_img(iloc,jloc) = 0;
			else
			  next_img(iloc,jloc) = couleur;
			if (current_img != next_img(iloc,jloc))
			  {
			    stop_it = 0;
			    stop_tuile = 0;
			  }
		      }
		  next[i][j] = stop_tuile;
	      
		}
	    }
      }
      swap_images();
      
      int** tmp;
      tmp = courant;
      courant = next;
      next = tmp; 
      
      
      if (stop_it)
	return it;
    }
  
  return 0; // on ne s'arrête jamais
}

///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v8 (unsigned nb_iter)
{
  return ocl_compute (nb_iter);
}
