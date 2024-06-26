/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:08:04 GMT 2023
 */

package org.apache.commons.math3.genetics;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import org.apache.commons.math3.genetics.Chromosome;
import org.apache.commons.math3.genetics.ElitisticListPopulation;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ElitisticListPopulation_ESTest extends ElitisticListPopulation_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      LinkedList<Chromosome> linkedList0 = new LinkedList<Chromosome>();
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(linkedList0, 270, 270);
      ElitisticListPopulation elitisticListPopulation1 = (ElitisticListPopulation)elitisticListPopulation0.nextGeneration();
      assertNotSame(elitisticListPopulation1, elitisticListPopulation0);
      assertEquals(270.0, elitisticListPopulation1.getElitismRate(), 0.01);
      assertEquals(270.0, elitisticListPopulation0.getElitismRate(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      LinkedList<Chromosome> linkedList0 = new LinkedList<Chromosome>();
      linkedList0.add((Chromosome) null);
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(linkedList0, 270, 270);
      elitisticListPopulation0.addChromosome((Chromosome) null);
      linkedList0.remove();
      // Undeclared exception!
      try { 
        elitisticListPopulation0.nextGeneration();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: -269, Size: 1
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(1160, 803.1);
      // Undeclared exception!
      try { 
        elitisticListPopulation0.setElitismRate((-22.56284143157783));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // elitism rate (-22.563)
         //
         verifyException("org.apache.commons.math3.genetics.ElitisticListPopulation", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(1160, 803.1);
      elitisticListPopulation0.setElitismRate(0.0);
      assertEquals(0.0, elitisticListPopulation0.getElitismRate(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(2760, 3745.2454230146);
      // Undeclared exception!
      try { 
        elitisticListPopulation0.setElitismRate(2760);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // elitism rate (2,760)
         //
         verifyException("org.apache.commons.math3.genetics.ElitisticListPopulation", e);
      }
  }
}
