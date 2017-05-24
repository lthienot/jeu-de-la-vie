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
unsigned compute_v9 (unsigned nb_iter);
unsigned compute_v10 (unsigned nb_iter);

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
  NULL,
  NULL
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
  NULL,
  NULL
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
  compute_v9,
  compute_v10
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
  "OpenCL naïve",
  "OpenCL optimisée",
  "OpenCL très optimisée"
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
  1,
  1,
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
      int img_stable = 1;
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
	      img_stable = 0;
	  }
      swap_images();

      if (img_stable)
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
      int img_stable = 1;
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
		  img_stable = 0;
	      }
      swap_images();
      if (img_stable)
	return it;
    }

  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}

///////////////////////////// Version séquentielle tuilée optimisée

int** mvt_suivant;
int** mvt_courant;
int** bords_courant;
int** bords_suivant;

void init_v2()
{
  mvt_suivant = malloc(TILE_NUMBER*sizeof(int*));
  mvt_courant = malloc(TILE_NUMBER*sizeof(int*));
  bords_suivant = malloc(TILE_NUMBER*sizeof(int*));
  bords_courant = malloc(TILE_NUMBER*sizeof(int*));
  for (int y = 0; y < TILE_NUMBER; y++)
    {
      mvt_suivant[y] = malloc(TILE_NUMBER*sizeof(int));
      mvt_courant[y] = malloc(TILE_NUMBER*sizeof(int));
      bords_suivant[y] = malloc(TILE_NUMBER*sizeof(int));
      bords_courant[y] = malloc(TILE_NUMBER*sizeof(int));
      for (int z = 0; z < TILE_NUMBER; z++)
	{
	  mvt_courant[y][z] = 0;
	  mvt_suivant[y][z] = 0;
	  bords_courant[y][z] = 0;
	  bords_suivant[y][z] = 0;
	}
    }
}

unsigned compute_v2 (unsigned nb_iter) //ça marche pas !!!!
{
  int img_stable;

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      img_stable = 1;

      for (int i = 0; i < TILE_NUMBER; i++)
	for (int j = 0; j < TILE_NUMBER; j++)
	  {
	    if (mvt_courant[i][j] == 0 ||
		(mvt_courant[i][j] == 1 &&
		 ((i>0 && bords_courant[i-1][j] == 0) ||
		  (i>0 && j>0 && bords_courant[i-1][j-1] == 0) ||
		  (i>0 && j<TILE_NUMBER-1 && bords_courant[i-1][j+1] == 0) ||
		  (i<TILE_NUMBER-1 && bords_courant[i+1][j] == 0) ||
		  (i<TILE_NUMBER-1 && j>0 && bords_courant[i+1][j-1] == 0) ||
		  (i<TILE_NUMBER-1 && j<TILE_NUMBER-1 && bords_courant[i+1][j+1] == 0) ||
		  (j>0 && bords_courant[i][j-1] == 0) ||
		  (j<TILE_NUMBER-1 && bords_courant[i][j+1] == 0))))
	      {

		int tuile_stable = 1;
		int bords_stables = 1;
		for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
		  for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
		    {
		      int count = count_neighbours(iloc,jloc);
		      int current_img = cur_img(iloc,jloc);
		      if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
			{
			  next_img(iloc,jloc) = 0;
			  if (current_img != 0 && (iloc % TILE_SIZE == 0 || jloc % TILE_SIZE == 0 || (iloc+1) % TILE_SIZE == 0 || (jloc+1) % TILE_SIZE == 0)) bords_stables = 0;
			}
		      else
			{
			  next_img(iloc,jloc) = couleur;
			  if (current_img == 0 && (iloc % TILE_SIZE == 0 || jloc % TILE_SIZE == 0 || (iloc+1) % TILE_SIZE == 0 || (jloc+1) % TILE_SIZE == 0)) bords_stables = 0;
			}
		      if (current_img != next_img(iloc,jloc))
    			{
    			  img_stable = 0;
    			  tuile_stable = 0;
    			}
		    }
		mvt_suivant[i][j] = tuile_stable;
		bords_suivant[i][j] = bords_stables;
	      }
	  }
      if (img_stable)
	return it;
      swap_images();

      int** tmp;
      tmp = mvt_courant;
      mvt_courant = mvt_suivant;
      mvt_suivant = tmp;
      tmp = bords_courant;
      bords_courant = bords_suivant;
      bords_suivant = tmp;
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
      int img_stable = 1;
#pragma omp parallel for collapse(2) schedule(static, 1)
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
	      img_stable = 0;
	  }
      swap_images();

      if (img_stable)
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
      int img_stable = 1;
#pragma omp parallel for collapse(2) schedule(static, 1)
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
	  	  img_stable = 0;
	      }
      swap_images();
      if (img_stable)
	return it;
    }

  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenMP optimisée for


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v5(unsigned nb_iter)
{
  int img_stable;
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      img_stable = 1;
#pragma omp parallel for collapse(2) schedule(static, 1)
      for (int i = 0; i < TILE_NUMBER; i++)
	for (int j = 0; j < TILE_NUMBER; j++)
	  {
	    if (mvt_courant[i][j] == 0 ||
		(mvt_courant[i][j] == 1 &&
		 ((i>0 && bords_courant[i-1][j] == 0) ||
		  (i>0 && j>0 && bords_courant[i-1][j-1] == 0) ||
		  (i>0 && j<TILE_NUMBER-1 && bords_courant[i-1][j+1] == 0) ||
		  (i<TILE_NUMBER-1 && bords_courant[i+1][j] == 0) ||
		  (i<TILE_NUMBER-1 && j>0 && bords_courant[i+1][j-1] == 0) ||
		  (i<TILE_NUMBER-1 && j<TILE_NUMBER-1 && bords_courant[i+1][j+1] == 0) ||
		  (j>0 && bords_courant[i][j-1] == 0) ||
		  (j<TILE_NUMBER-1 && bords_courant[i][j+1] == 0))))
	      {
		int tuile_stable = 1;
		int bords_stables = 1;
		for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
		  for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
		    {
		      int count = count_neighbours(iloc,jloc);
		      int current_img = cur_img(iloc,jloc);
		      if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
			{
			  next_img(iloc,jloc) = 0;
			  if (current_img != 0 && (iloc % TILE_SIZE == 0 || jloc % TILE_SIZE == 0 || (iloc+1) % TILE_SIZE == 0 || (jloc+1) % TILE_SIZE == 0)) bords_stables = 0;
			}
		      else
			{
			  next_img(iloc,jloc) = couleur;
			  if (current_img == 0 && (iloc % TILE_SIZE == 0 || jloc % TILE_SIZE == 0 || (iloc+1) % TILE_SIZE == 0 || (jloc+1) % TILE_SIZE == 0)) bords_stables = 0;
			}
		      if (current_img != next_img(iloc,jloc))
    			{
    			  img_stable = 0;
    			  tuile_stable = 0;
    			}
		    }
		mvt_suivant[i][j] = tuile_stable;
		bords_suivant[i][j] = bords_stables;
	      }
	  }
      if (img_stable)
	return it;
      swap_images();

      int** tmp;
      tmp = mvt_courant;
      mvt_courant = mvt_suivant;
      mvt_suivant = tmp;
      tmp = bords_courant;
      bords_courant = bords_suivant;
      bords_suivant = tmp;
    }

  return 0; // on ne s'arrête jamais
}

///////////////////////////// Version OpenMP tuilée task



// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v6(unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int img_stable = 1;
#pragma omp parallel shared(img_stable)
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
		      img_stable = 0;
		  }
	    }
      }

      if (img_stable)
	return it;
      swap_images();
    }
  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenMP optimisée task


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v7(unsigned nb_iter)
{
  int img_stable;
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      img_stable = 1;
#pragma omp parallel
#pragma omp single
      {
	for (int i = 0; i < TILE_NUMBER; i++)
	  for (int j = 0; j < TILE_NUMBER; j++)
	    {
#pragma omp task firstprivate(i,j)
	      if (mvt_courant[i][j] == 0 ||
		  (mvt_courant[i][j] == 1 &&
		   ((i>0 && bords_courant[i-1][j] == 0) ||
		    (i>0 && j>0 && bords_courant[i-1][j-1] == 0) ||
		    (i>0 && j<TILE_NUMBER-1 && bords_courant[i-1][j+1] == 0) ||
		    (i<TILE_NUMBER-1 && bords_courant[i+1][j] == 0) ||
		    (i<TILE_NUMBER-1 && j>0 && bords_courant[i+1][j-1] == 0) ||
		    (i<TILE_NUMBER-1 && j<TILE_NUMBER-1 && bords_courant[i+1][j+1] == 0) ||
		    (j>0 && bords_courant[i][j-1] == 0) ||
		    (j<TILE_NUMBER-1 && bords_courant[i][j+1] == 0))))
		{
		  int tuile_stable = 1;
		  int bords_stables = 1;
		  for(int iloc = i*TILE_SIZE; iloc < (i+1)*TILE_SIZE && iloc < DIM; iloc++)
		    for(int jloc = j*TILE_SIZE; jloc < (j+1)*TILE_SIZE && jloc < DIM; jloc++)
		      {
			int count = count_neighbours(iloc,jloc);
			int current_img = cur_img(iloc,jloc);
			if ((current_img && (count < 2 || count > 3)) || (current_img == 0 && (count != 3)))
			  {
			    next_img(iloc,jloc) = 0;
			    if (current_img != 0 && (iloc % TILE_SIZE == 0 || jloc % TILE_SIZE == 0 || (iloc+1) % TILE_SIZE == 0 || (jloc+1) % TILE_SIZE == 0)) bords_stables = 0;
			  }
			else
			  {
			    next_img(iloc,jloc) = couleur;
			    if (current_img == 0 && (iloc % TILE_SIZE == 0 || jloc % TILE_SIZE == 0 || (iloc+1) % TILE_SIZE == 0 || (jloc+1) % TILE_SIZE == 0)) bords_stables = 0;
			  }
			if (current_img != next_img(iloc,jloc))
			  {
			    img_stable = 0;
			    tuile_stable = 0;
			  }
		      }
		  mvt_suivant[i][j] = tuile_stable;
		  bords_suivant[i][j] = bords_stables;
		}
	    }
      }
      if (img_stable)
	return it;
      swap_images();

      int** tmp;
      tmp = mvt_courant;
      mvt_courant = mvt_suivant;
      mvt_suivant = tmp;
      tmp = bords_courant;
      bords_courant = bords_suivant;
      bords_suivant = tmp;
    }

  return 0; // on ne s'arrête jamais
}

///////////////////////////// Version OpenCL naive

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v8 (unsigned nb_iter)
{
  return ocl_compute_naif(nb_iter);
}

///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v9 (unsigned nb_iter)
{
  return ocl_compute(nb_iter);
}

///////////////////////////// Version OpenCL Optimisée

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v10 (unsigned nb_iter)
{
  return ocl_compute_stop(nb_iter);
}
