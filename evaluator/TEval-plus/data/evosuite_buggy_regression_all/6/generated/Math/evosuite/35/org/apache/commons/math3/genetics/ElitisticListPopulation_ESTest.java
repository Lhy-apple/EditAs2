/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:49:29 GMT 2023
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
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(linkedList0, 237, 1700);
      ElitisticListPopulation elitisticListPopulation1 = (ElitisticListPopulation)elitisticListPopulation0.nextGeneration();
      assertNotSame(elitisticListPopulation1, elitisticListPopulation0);
      assertEquals(1700.0, elitisticListPopulation0.getElitismRate(), 0.01);
      assertEquals(1700.0, elitisticListPopulation1.getElitismRate(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      LinkedList<Chromosome> linkedList0 = new LinkedList<Chromosome>();
      linkedList0.offerLast((Chromosome) null);
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(linkedList0, 237, 1700);
      // Undeclared exception!
      try { 
        elitisticListPopulation0.nextGeneration();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: -1699, Size: 1
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(781, (-1210.0));
      // Undeclared exception!
      try { 
        elitisticListPopulation0.setElitismRate((-1210.0));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // elitism rate (-1,210)
         //
         verifyException("org.apache.commons.math3.genetics.ElitisticListPopulation", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(134, 134);
      elitisticListPopulation0.setElitismRate(0.0);
      assertEquals(0.0, elitisticListPopulation0.getElitismRate(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ElitisticListPopulation elitisticListPopulation0 = new ElitisticListPopulation(134, 134);
      // Undeclared exception!
      try { 
        elitisticListPopulation0.setElitismRate(134.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // elitism rate (134)
         //
         verifyException("org.apache.commons.math3.genetics.ElitisticListPopulation", e);
      }
  }
}
