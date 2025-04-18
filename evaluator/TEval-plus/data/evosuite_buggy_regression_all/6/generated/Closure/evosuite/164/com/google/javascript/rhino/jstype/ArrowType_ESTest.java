/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:08:01 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.ArrowType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.Visitor;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArrowType_ESTest extends ArrowType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "v Li2,?,%rGm");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      // Undeclared exception!
      try { 
        arrowType0.testForEquality(errorFunctionType0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.ArrowType", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "xU#L*z4c");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      // Undeclared exception!
      try { 
        arrowType0.getGreatestSubtype(errorFunctionType0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.ArrowType", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      // Undeclared exception!
      try { 
        arrowType0.getLeastSupertype(errorFunctionType0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.ArrowType", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      // Undeclared exception!
      try { 
        arrowType0.toString();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      // Undeclared exception!
      try { 
        arrowType0.visit((Visitor<Object>) null);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.ArrowType", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "izR");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      arrowType0.getPossibleToBooleanOutcomes();
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType((Node) null, arrowType0);
      boolean boolean0 = arrowType1.isSubtype(arrowType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Fc2W");
      ArrowType arrowType1 = errorFunctionType0.getInternalArrowType();
      boolean boolean0 = arrowType0.isSubtype(arrowType1);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "<&#Z)>{B]`Gah");
      Node node0 = Node.newString(1, "Not declared as a constructor", 1, 0);
      Node node1 = new Node(1, node0, 2, 2832);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node1, errorFunctionType0);
      boolean boolean0 = arrowType0.isSubtype(arrowType0);
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      Node node0 = errorFunctionType0.getParametersNode();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      Node node1 = new Node(38, node0, 23, 8232);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType(node1, (JSType) null);
      boolean boolean0 = arrowType0.isSubtype(arrowType1);
      assertFalse(errorFunctionType0.hasCachedValues());
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      boolean boolean0 = arrowType0.isSubtype(arrowType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "<&#Z)>{B]`Gah");
      Node node0 = Node.newString(1, "Not declared as a constructor", 1, 0);
      Node node1 = new Node(1, node0, 2, 2832);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node1, errorFunctionType0);
      boolean boolean0 = arrowType0.hasEqualParameters(arrowType0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      Node node0 = errorFunctionType0.getParametersNode();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      Node node1 = new Node((-1173), node0, 42, 0);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType(node1, arrowType0);
      boolean boolean0 = arrowType0.hasEqualParameters(arrowType1);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      boolean boolean0 = arrowType0.equals(arrowType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      Node node0 = errorFunctionType0.getParametersNode();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType((Node) null, arrowType0);
      boolean boolean0 = arrowType0.hasEqualParameters(arrowType1);
      assertFalse(boolean0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "<&#Z)>{B]`Gah");
      Node node0 = errorFunctionType0.getParametersNode();
      Node node1 = new Node(1, node0, 2, 2832);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node1, errorFunctionType0);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType(node0, errorFunctionType0);
      boolean boolean0 = arrowType0.hasEqualParameters(arrowType1);
      assertFalse(boolean0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "String node not created with Node.newString");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      boolean boolean0 = arrowType0.isEquivalentTo(errorFunctionType0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      ArrowType arrowType1 = new ArrowType(jSTypeRegistry0, (Node) null, arrowType0);
      boolean boolean0 = arrowType0.equals(arrowType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = errorFunctionType0.getParametersNode();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      arrowType0.returnType = (JSType) null;
      arrowType0.hashCode();
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertEquals(3, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      ArrowType arrowType1 = new ArrowType(jSTypeRegistry0, (Node) null, arrowType0, true);
      arrowType1.hashCode();
      assertFalse(arrowType1.equals((Object)arrowType0));
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      JSType[] jSTypeArray0 = new JSType[0];
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs(jSTypeArray0);
      Node node1 = new Node(1, node0, 1, 46);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node1);
      arrowType0.hashCode();
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = errorFunctionType0.getParametersNode();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      arrowType0.resolveInternal(simpleErrorReporter0, errorFunctionType0);
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(node0.hasMoreThanOneChild());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = errorFunctionType0.getParametersNode();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(errorFunctionType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) linkedList0);
      Node node1 = new Node((-1658), node0);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node1);
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null);
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = errorFunctionType0.getParametersNode();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      arrowType0.returnType = (JSType) null;
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertTrue(boolean0);
      assertEquals(3, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) linkedList0);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node0);
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(boolean0);
  }
}
