/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:08:35 GMT 2023
 */

package org.apache.commons.math.optimization.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import org.apache.commons.math.linear.ArrayRealVector;
import org.apache.commons.math.linear.OpenMapRealVector;
import org.apache.commons.math.optimization.GoalType;
import org.apache.commons.math.optimization.RealPointValuePair;
import org.apache.commons.math.optimization.linear.LinearConstraint;
import org.apache.commons.math.optimization.linear.LinearObjectiveFunction;
import org.apache.commons.math.optimization.linear.Relationship;
import org.apache.commons.math.optimization.linear.SimplexTableau;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SimplexTableau_ESTest extends SimplexTableau_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      double[][] doubleArray0 = simplexTableau0.getData();
      assertEquals(1, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      ArrayRealVector arrayRealVector0 = new ArrayRealVector(doubleArray0, doubleArray0);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(arrayRealVector0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, (-1773.7858362524848));
      // Undeclared exception!
      try { 
        simplexTableau0.setEntry((-1), (-1), 1509.31893142);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // no entry at indices (-1, -1) in a 1x5 matrix
         //
         verifyException("org.apache.commons.math.linear.Array2DRowRealMatrix", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      double[] doubleArray0 = new double[11];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 2.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 2.0);
      RealPointValuePair realPointValuePair0 = simplexTableau0.getSolution();
      assertEquals(2.0, realPointValuePair0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 2.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      Relationship relationship0 = Relationship.EQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, relationship0, 2.0);
      linkedList0.add(linearConstraint0);
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 2.0);
      RealPointValuePair realPointValuePair0 = simplexTableau0.getSolution();
      assertEquals(2.0, realPointValuePair0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-1488.6574242768));
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 0.0);
      int int0 = simplexTableau0.getNumDecisionVariables();
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      int int0 = simplexTableau0.getNumSlackVariables();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      int int0 = simplexTableau0.getNumArtificialVariables();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-450.21));
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 0.0);
      simplexTableau0.hashCode();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[10];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 2.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      Relationship relationship0 = Relationship.GEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(openMapRealVector0, relationship0, 2297.639248623);
      linkedList0.addLast(linearConstraint0);
      linkedList0.add(linearConstraint0);
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 2.0);
      assertEquals(10, simplexTableau0.getNumVariables());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      double[] doubleArray0 = new double[0];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-1102.8));
      double[] doubleArray1 = new double[3];
      doubleArray1[1] = (-1102.8);
      Relationship relationship0 = Relationship.GEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray1, relationship0, (-1102.8));
      linkedList0.offerFirst(linearConstraint0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = null;
      try {
        simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, (-1102.8));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.linear.SimplexTableau", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      simplexTableau0.discardArtificialVariables();
      assertEquals(171, simplexTableau0.getNumVariables());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 1.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      Relationship relationship0 = Relationship.GEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, 1.0, relationship0, doubleArray0, (-10.335849158441771));
      linkedList0.add(linearConstraint0);
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 3.141592653589793);
      simplexTableau0.discardArtificialVariables();
      assertEquals(5, simplexTableau0.getNumVariables());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, (-1488.6574242768));
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 0.0);
      simplexTableau0.divideRow(0, 0);
      assertEquals(5, simplexTableau0.getNumVariables());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 2.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 2.0);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 2.0);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector();
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      boolean boolean0 = simplexTableau0.equals(simplexTableau0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(139);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 1.0E-12);
      boolean boolean0 = simplexTableau0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171, 171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 171);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0E-12);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0E-12);
      double[] doubleArray0 = new double[1];
      LinearObjectiveFunction linearObjectiveFunction1 = new LinearObjectiveFunction(doubleArray0, 1.0E-12);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction1, linkedList0, goalType0, false, (-1355.0836682));
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      Relationship relationship0 = Relationship.LEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(openMapRealVector0, relationship0, 171);
      linkedList0.add(linearConstraint0);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(doubleArray0, 2.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 2.0);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 2.0);
      assertTrue(simplexTableau1.equals((Object)simplexTableau0));
      
      simplexTableau1.numArtificialVariables = 1633;
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171, 171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 171);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 1.0E-12);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171, 171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 0.0);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      LinearObjectiveFunction linearObjectiveFunction1 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction1, linkedList0, goalType0, false, (-1.0));
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, (-1.0));
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(171);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      LinkedList<LinearConstraint> linkedList1 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList1, goalType0, false, 1.0);
      double[] doubleArray0 = new double[2];
      Relationship relationship0 = Relationship.GEQ;
      LinearConstraint linearConstraint0 = new LinearConstraint(doubleArray0, relationship0, 171);
      linkedList1.addFirst(linearConstraint0);
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, false, 1.0);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(156);
      LinearObjectiveFunction linearObjectiveFunction0 = new LinearObjectiveFunction(openMapRealVector0, 1.0E-12);
      LinkedList<LinearConstraint> linkedList0 = new LinkedList<LinearConstraint>();
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplexTableau simplexTableau0 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType0, true, 1.0);
      GoalType goalType1 = GoalType.MINIMIZE;
      SimplexTableau simplexTableau1 = new SimplexTableau(linearObjectiveFunction0, linkedList0, goalType1, true, 1.0);
      boolean boolean0 = simplexTableau0.equals(simplexTableau1);
      assertFalse(boolean0);
  }
}
