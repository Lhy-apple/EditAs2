/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:39:43 GMT 2023
 */

package org.apache.commons.math3.optimization.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optimization.GoalType;
import org.apache.commons.math3.optimization.PointValuePair;
import org.apache.commons.math3.optimization.linear.LinearConstraint;
import org.apache.commons.math3.optimization.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optimization.linear.Relationship;
import org.apache.commons.math3.optimization.linear.SimplexTableau;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SimplexTableau_ESTest extends SimplexTableau_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 739);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 739);
      // Undeclared exception!
      try { 
        simplexTableau0.setEntry(739, 739, 739);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // row index (739)
         //
         verifyException("org.apache.commons.math3.linear.MatrixUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, (-1.0));
      // Undeclared exception!
      try { 
        simplexTableau0.subtractRow(1987, 1987, (-1.0));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // row index (1,987)
         //
         verifyException("org.apache.commons.math3.linear.MatrixUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, (-855.8592), 0);
      double[][] doubleArray1 = simplexTableau0.getData();
      assertEquals(1, doubleArray1.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-1.0);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-1.0));
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.1677812418806031E-60);
      PointValuePair pointValuePair0 = simplexTableau0.getSolution();
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, (-855.8592), 0);
      simplexTableau0.hashCode();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 1077.0613373551);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      int int0 = simplexTableau0.getNumDecisionVariables();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[5] = (-1.0);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      Relationship relationship0 = Relationship.GEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, relationship0, (-311.503289987));
      linkedList0.push(linearConstraint0);
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, (-1.0));
      PointValuePair pointValuePair0 = simplexTableau0.getSolution();
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 0.0);
      simplexTableau0.dropPhase1Objective();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 3889.7526853268);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      Relationship relationship0 = Relationship.EQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, 1.0, relationship0, doubleArray0, 1.0);
      linkedList0.add(linearConstraint0);
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 3889.7526853268, (-2712));
      simplexTableau0.dropPhase1Objective();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 0.0);
      boolean boolean0 = simplexTableau0.isOptimal();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[0] = 3114.665361305665;
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-1293.82428));
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, (-1369.14520325));
      boolean boolean0 = simplexTableau0.isOptimal();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      Relationship relationship0 = Relationship.GEQ;
      double[] doubleArray0 = new double[7];
      doubleArray0[4] = (-1.0);
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, relationship0, 61.0119698157769);
      linkedList0.add(linearConstraint0);
      GoalType goalType0 = GoalType.MINIMIZE;
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 1.0);
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 61.0119698157769);
      simplexTableau0.dropPhase1Objective();
      PointValuePair pointValuePair0 = simplexTableau0.getSolution();
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-1.0);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-2230.6591121155));
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, (-1.0));
      PointValuePair pointValuePair0 = simplexTableau0.getSolution();
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      double[] doubleArray0 = new double[1];
      Relationship relationship0 = Relationship.LEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, (-4.869955151949929E-231), relationship0, doubleArray0, (-8.221241452580445E-8));
      linkedList0.offerLast(linearConstraint0);
      doubleArray0[0] = (-1.0);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-2230.6591121155));
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, (-1.0));
      PointValuePair pointValuePair0 = simplexTableau0.getSolution();
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(741);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 741, 0);
      simplexTableau0.divideRow(0, 1377.5334103187);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(758);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 758);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 758);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 758);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 739);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 739);
      boolean boolean0 = simplexTableau0.equals(simplexTableau0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(739);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 739);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 2515.69538695149, (-1));
      boolean boolean0 = simplexTableau0.equals(arrayRealVector0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(758);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 758);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 758);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 758);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(739);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 739);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 2515.69538695149, (-1));
      double[] doubleArray0 = new double[1];
      LinearObjectiveFunction linearObjectiveFunction1 = new LinearObjectiveFunction(doubleArray0, (-1));
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction1, linkedList0, goalType0, false, 2.1842007637023926, 212);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(758);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 758);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 758);
      Relationship relationship0 = Relationship.LEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(arrayRealVector0, (-1295.590140091), relationship0, arrayRealVector0, 739.1250253021717);
      linkedList0.add(linearConstraint0);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 758);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 3889.7526853268);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      Relationship relationship0 = Relationship.EQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, 1.0, relationship0, doubleArray0, 1.0);
      linkedList0.add(linearConstraint0);
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 3889.7526853268);
      simplexTableau0.dropPhase1Objective();
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, (-1886.394));
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 773.1441386089697);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 739);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 773.1441386089697);
      boolean boolean0 = simplexTableau1.equals(simplexTableau0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector();
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 739);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 739);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 739, 739);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
      assertFalse(simplexTableau1.equals((Object)simplexTableau0));
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(741);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 741);
      LinearObjectiveFunction linearObjectiveFunction1 = new LinearObjectiveFunction(arrayRealVector0, 3888.517611);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 741);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction1, linkedList0, goalType0, false, 741);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(739);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 739);
      GoalType goalType1 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType1, false, 739);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }
}