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
  compute_v8
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
  1,
};

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
	    if (cur_img(i,j))
	      {
		if(count < 2 || count > 3)
		  next_img(i,j) = 0;
		else
		  next_img(i,j) = cur_img(i,j);
	      }
	    else
	      {
		if (count != 3)
		  next_img(i,j) = 0;
		else
		  next_img(i,j) = couleur;
	      }
	    if (cur_img(i,j)!=next_img(i,j))
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

#define GRAIN 32

unsigned compute_v1 (unsigned nb_iter)
{
  int tranche = DIM/GRAIN;

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
      for (int i = 0; i < GRAIN; i++)
	for (int j = 0; j < GRAIN; j++)
	  for(int iloc = i*tranche; iloc < (i+1)*tranche && iloc < DIM; iloc++)
	    for(int jloc = j*tranche; jloc < (j+1)*tranche && jloc < DIM; jloc++)
	      {
		int count = 0;
		if(iloc>0 && jloc>0 && cur_img(iloc-1,jloc-1))
		  count++;
		if(jloc>0 && cur_img(iloc,jloc-1))
		  count++;
		if(iloc<DIM-1 && jloc>0 && cur_img(iloc+1,jloc-1))
		  count++;
		if(iloc>0 && cur_img(iloc-1,jloc))
		  count++;
		if(iloc<DIM-1 && cur_img(iloc+1,jloc) )
		  count++;
		if(iloc>0 && jloc<DIM-1 && cur_img(iloc-1,jloc+1))
		  count++;
		if(jloc<DIM-1 && cur_img(iloc,jloc+1))
		  count++;
		if(iloc<DIM-1 && jloc<DIM-1 && cur_img(iloc+1,jloc+1) )
		  count++;
		if (cur_img(iloc,jloc))
		  {
		    if(count < 2 || count > 3)
		      next_img(iloc,jloc) = 0;
		    else
		      next_img(iloc,jloc) = cur_img(iloc,jloc);
		  }
		else
		  {
		    if (count != 3)
		      next_img(iloc,jloc) = 0;
		    else
		      next_img(iloc,jloc) = couleur;
		  }
		if (cur_img(iloc,jloc)!=next_img(iloc,jloc))
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
  next = malloc(GRAIN*sizeof(int*));
  courant = malloc(GRAIN*sizeof(int*));
  for (int y = 0; y < GRAIN; y++)
    {
      next[y] = malloc(GRAIN*sizeof(int));
      courant[y] = malloc(GRAIN*sizeof(int));
      for (int z = 0; z < GRAIN; z++)
	{
	  courant[y][z] = 0;
	  next[y][z] = 0;
	}
    }
}

unsigned compute_v2 (unsigned nb_iter) //ça marche pas !!!!
{
  int tranche = DIM/GRAIN;
  int stop_it;
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      stop_it = 1;
      
      for (int i = 0; i < GRAIN; i++)
	for (int j = 0; j < GRAIN; j++)
	  {
	    if (courant[i][j] == 0 ||
		(courant[i][j] == 1 &&
		 ((i>0 && courant[i-1][j] == 0) ||
		  (i>0 && j>0 && courant[i-1][j-1] == 0) ||
		  (i>0 && j<GRAIN-1 && courant[i-1][j+1] == 0) ||
		  (i<GRAIN-1 && courant[i+1][j] == 0) ||
		  (i<GRAIN-1 && j>0 && courant[i+1][j-1] == 0) ||
		  (i<GRAIN-1 && j<GRAIN-1 && courant[i+1][j+1] == 0) ||
		  (j>0 && courant[i][j-1] == 0) ||
		  (j<GRAIN-1 && courant[i][j+1] == 0))))
	      {

		int stop_tuile = 1;
		for(int iloc = i*tranche; iloc < (i+1)*tranche && iloc < DIM; iloc++)
		  for(int jloc = j*tranche; jloc < (j+1)*tranche && jloc < DIM; jloc++)
		    {
		      int count = 0;
		      if(iloc>0 && jloc>0 && cur_img(iloc-1,jloc-1))
			count++;
		      if(jloc>0 && cur_img(iloc,jloc-1))
			count++;
		      if(iloc<DIM-1 && jloc>0 && cur_img(iloc+1,jloc-1))
			count++;
		      if(iloc>0 && cur_img(iloc-1,jloc))
			count++;
		      if(iloc<DIM-1 && cur_img(iloc+1,jloc) )
			count++;
		      if(iloc>0 && jloc<DIM-1 && cur_img(iloc-1,jloc+1))
			count++;
		      if(jloc<DIM-1 && cur_img(iloc,jloc+1))
			count++;
		      if(iloc<DIM-1 && jloc<DIM-1 && cur_img(iloc+1,jloc+1) )
			count++;
		      if (cur_img(iloc,jloc))
			{
			  if(count < 2 || count > 3)
			    next_img(iloc,jloc) = 0;
			  else
			    next_img(iloc,jloc) = cur_img(iloc,jloc);
			}
		      else
			{
			  if (count != 3)
			    next_img(iloc,jloc) = 0;
			  else
			    next_img(iloc,jloc) = couleur;
			}
		      if (cur_img(iloc,jloc) != next_img(iloc,jloc))
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
	    if (cur_img(i,j))
	      {
		if(count < 2 || count > 3)
		  next_img(i,j) = 0;
		else
		  next_img(i,j) = cur_img(i,j);
	      }
	    else
	      {
		if (count != 3)
		  next_img(i,j) = 0;
		else
		  next_img(i,j) = couleur;
	      }
	    if (cur_img(i,j)!=next_img(i,j))
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
  int tranche = DIM/GRAIN;

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
#pragma omp parallel for collapse(2)
      /* #pragma omp parallel for collapse(4) */ /* version qui marche bien mais pas autant que le collapse 2 */
      for (int i = 0; i < GRAIN; i++)
	for (int j = 0; j < GRAIN; j++)
	  /* for(int iloc1 = 0; iloc1 < tranche; iloc1++) */
	  /*   for(int jloc1 = 0; jloc1 < tranche; jloc1++) */
	  /*     { */
	  /* 	int iloc = i*tranche+iloc1; */
	  /* 	int jloc = j*tranche+jloc1; */
	  /* 	int count = 0; */
	  /* 	if(iloc < DIM && jloc < DIM) */
	  /* 	  { */
	  /* 	    if(iloc>0 && jloc>0 && cur_img(iloc-1,jloc-1)) */
	  /* 	      count++; */
	  /* 	    if(jloc>0 && cur_img(iloc,jloc-1)) */
	  /* 	      count++; */
	  /* 	    if(iloc<DIM-1 && jloc>0 && cur_img(iloc+1,jloc-1)) */
	  /* 	      count++; */
	  /* 	    if(iloc>0 && cur_img(iloc-1,jloc)) */
	  /* 	      count++; */
	  /* 	    if(iloc<DIM-1 && cur_img(iloc+1,jloc) ) */
	  /* 	      count++; */
	  /* 	    if(iloc>0 && jloc<DIM-1 && cur_img(iloc-1,jloc+1)) */
	  /* 	      count++; */
	  /* 	    if(jloc<DIM-1 && cur_img(iloc,jloc+1)) */
	  /* 	      count++; */
	  /* 	    if(iloc<DIM-1 && jloc<DIM-1 && cur_img(iloc+1,jloc+1) ) */
	  /* 	      count++; */
	  /* 	    if (cur_img(iloc,jloc)) */
	  /* 	      if(count < 2 || count > 3) */
	  /* 		next_img(iloc,jloc) = 0; */
	  /* 	      else */
	  /* 		next_img(iloc,jloc) = cur_img(iloc,jloc); */
	  /* 	    else */
	  /* 	      if (count != 3) */
	  /* 		next_img(iloc,jloc) = 0; */
	  /* 	      else */
	  /* 		next_img(iloc,jloc) = couleur; */
	  /* 	    if (cur_img(iloc,jloc)!=next_img(iloc,jloc)) */
	  /* 	      stop_it = 0; */
	  /* 	  } */
	  /*     } */
	  for(int iloc = i*tranche; iloc < (i+1)*tranche && iloc < DIM; iloc++)
	    for(int jloc = j*tranche; jloc < (j+1)*tranche && jloc < DIM; jloc++)
	      {
		int count = 0;
		if(iloc>0 && jloc>0 && cur_img(iloc-1,jloc-1))
	  	  count++;
	  	if(jloc>0 && cur_img(iloc,jloc-1))
	  	  count++;
	  	if(iloc<DIM-1 && jloc>0 && cur_img(iloc+1,jloc-1))
	  	  count++;
	  	if(iloc>0 && cur_img(iloc-1,jloc))
	  	  count++;
	  	if(iloc<DIM-1 && cur_img(iloc+1,jloc) )
	  	  count++;
	  	if(iloc>0 && jloc<DIM-1 && cur_img(iloc-1,jloc+1))
	  	  count++;
	  	if(jloc<DIM-1 && cur_img(iloc,jloc+1))
	  	  count++;
	  	if(iloc<DIM-1 && jloc<DIM-1 && cur_img(iloc+1,jloc+1) )
	  	  count++;
	  	if (cur_img(iloc,jloc))
		  {
		    if(count < 2 || count > 3)
		      next_img(iloc,jloc) = 0;
		    else
		      next_img(iloc,jloc) = cur_img(iloc,jloc);
		  }
	  	else
		  {
		    if (count != 3)
		      next_img(iloc,jloc) = 0;
		    else
		      next_img(iloc,jloc) = couleur;
		  }
	  	if (cur_img(iloc,jloc)!=next_img(iloc,jloc))
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
  int tranche = DIM/GRAIN;
  int stop_it;
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      stop_it = 1;
#pragma omp parallel for collapse(2)
      for (int i = 0; i < GRAIN; i++)
	for (int j = 0; j < GRAIN; j++)
	  {
	    if (courant[i][j] == 0 ||
		(courant[i][j] == 1 &&
		 ((i>0 && courant[i-1][j] == 0) ||
		  (i>0 && j>0 && courant[i-1][j-1] == 0) ||
		  (i>0 && j<GRAIN-1 && courant[i-1][j+1] == 0) ||
		  (i<GRAIN-1 && courant[i+1][j] == 0) ||
		  (i<GRAIN-1 && j>0 && courant[i+1][j-1] == 0) ||
		  (i<GRAIN-1 && j<GRAIN-1 && courant[i+1][j+1] == 0) ||
		  (j>0 && courant[i][j-1] == 0) ||
		  (j<GRAIN-1 && courant[i][j+1] == 0))))
	      {

		int stop_tuile = 1;
	      	for(int iloc = i*tranche; iloc < (i+1)*tranche && iloc < DIM; iloc++)
	      	  for(int jloc = j*tranche; jloc < (j+1)*tranche && jloc < DIM; jloc++)
	      	    {
	      	      int count = 0;
	      	      if(iloc>0 && jloc>0 && cur_img(iloc-1,jloc-1))
	      		count++;
	      	      if(jloc>0 && cur_img(iloc,jloc-1))
	      		count++;
	      	      if(iloc<DIM-1 && jloc>0 && cur_img(iloc+1,jloc-1))
	      		count++;
	      	      if(iloc>0 && cur_img(iloc-1,jloc))
	      		count++;
	      	      if(iloc<DIM-1 && cur_img(iloc+1,jloc) )
	      		count++;
	      	      if(iloc>0 && jloc<DIM-1 && cur_img(iloc-1,jloc+1))
	      		count++;
	      	      if(jloc<DIM-1 && cur_img(iloc,jloc+1))
	      		count++;
	      	      if(iloc<DIM-1 && jloc<DIM-1 && cur_img(iloc+1,jloc+1) )
	      		count++;
	      	      if (cur_img(iloc,jloc))
			{
			  if(count < 2 || count > 3)
			    next_img(iloc,jloc) = 0;
			  else
			    next_img(iloc,jloc) = cur_img(iloc,jloc);
			}
	      	      else
			{
			  if (count != 3)
			    next_img(iloc,jloc) = 0;
			  else
			    next_img(iloc,jloc) = couleur;
			}
	      	      if (cur_img(iloc,jloc) != next_img(iloc,jloc))
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
  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenMP optimisée task


// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v7(unsigned nb_iter)
{
  return 0; // on ne s'arrête jamais
}

///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v8 (unsigned nb_iter)
{
  return ocl_compute (nb_iter);
}
