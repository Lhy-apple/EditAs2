/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:59:53 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.ImmutableList;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.ArrowType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.IndexedType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ModificationVisitor;
import com.google.javascript.rhino.jstype.NoObjectType;
import com.google.javascript.rhino.jstype.NoType;
import com.google.javascript.rhino.jstype.NumberType;
import com.google.javascript.rhino.jstype.TemplateType;
import com.google.javascript.rhino.jstype.UnionType;
import com.google.javascript.rhino.jstype.VoidType;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArrowType_ESTest extends ArrowType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, " nm");
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
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "20|-U,{3~\"SP");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      // Undeclared exception!
      try { 
        arrowType0.getTypesUnderShallowEquality(errorFunctionType0);
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
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null, (JSType) null);
      // Undeclared exception!
      try { 
        arrowType0.getLeastSupertype((JSType) null);
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
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "20|-U,{3~\"SP");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      String string0 = arrowType0.toString();
      assertEquals("[ArrowType]", string0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "#hL");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      ImmutableList<JSType> immutableList0 = ImmutableList.of((JSType) arrowType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, immutableList0);
      // Undeclared exception!
      try { 
        modificationVisitor0.caseUnionType(unionType0);
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
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "r?SnA%Wt~U3h&Cf");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      arrowType0.getPossibleToBooleanOutcomes();
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "20|-U,{3~\"S");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      Node node0 = new Node(1);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType(node0, errorFunctionType0);
      boolean boolean0 = arrowType1.isSubtype(arrowType0);
      assertTrue(errorFunctionType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "2[|-U,{3~\"SkP");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      Node node0 = new Node(65536, 1, 0);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType(node0);
      boolean boolean0 = arrowType0.isSubtype(arrowType1);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertTrue(boolean0);
      assertFalse(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "2%>fD&WxBp\"5xqj");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      JSType[] jSTypeArray0 = new JSType[3];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType(node0);
      boolean boolean0 = arrowType1.isSubtype(arrowType0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "i|6(jK,!D32OWq[$");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      Node node0 = new Node(1);
      Node node1 = new Node(8, node0, node0, node0, node0, 43, 54);
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType(node1);
      boolean boolean0 = arrowType0.isSubtype(arrowType1);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(boolean0);
      assertFalse(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "!2[|N_-k,{3\"SkP");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      JSType[] jSTypeArray0 = new JSType[4];
      jSTypeArray0[0] = (JSType) noObjectType0;
      jSTypeArray0[1] = (JSType) errorFunctionType0;
      jSTypeArray0[2] = (JSType) arrowType0;
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      ArrowType arrowType1 = new ArrowType(jSTypeRegistry0, node0, (JSType) null);
      boolean boolean0 = arrowType0.isSubtype(arrowType1);
      assertFalse(boolean0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "AlZeF#0P{v1");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType((Node) null);
      boolean boolean0 = arrowType1.isSubtype(arrowType0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "H,(nx!pgXJ*");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      ArrowType arrowType1 = jSTypeRegistry0.createArrowType((Node) null);
      boolean boolean0 = arrowType0.isSubtype(arrowType1);
      assertTrue(boolean0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      JSType[] jSTypeArray0 = new JSType[3];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, noObjectType0);
      boolean boolean0 = arrowType0.isSubtype(arrowType0);
      assertTrue(noObjectType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null, (JSType) null);
      boolean boolean0 = arrowType0.isSubtype(arrowType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      VoidType voidType0 = new VoidType(jSTypeRegistry0);
      ImmutableList<JSType> immutableList0 = ImmutableList.of((JSType) voidType0, (JSType) voidType0, (JSType) voidType0, (JSType) voidType0, (JSType) voidType0);
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) immutableList0);
      Node node1 = new Node((-1), node0, node0, 49, 54);
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node1, voidType0, false);
      boolean boolean0 = arrowType0.hasEqualParameters(arrowType0, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "vAlZe0P{1");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      Node node0 = Node.newString("vAlZe0P{1");
      Node node1 = new Node((-1), node0, node0, node0);
      ArrowType arrowType1 = new ArrowType(jSTypeRegistry0, node1, errorFunctionType0);
      boolean boolean0 = arrowType0.hasEqualParameters(arrowType1, true);
      assertFalse(boolean0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null, (JSType) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Named type with empty name component");
      ArrowType arrowType1 = errorFunctionType0.getInternalArrowType();
      boolean boolean0 = arrowType0.hasEqualParameters(arrowType1, true);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "!)}3P");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      Node node0 = Node.newString(1038, "n$vGCy*be8:0p2mF&");
      Node node1 = new Node(52, node0, node0);
      ArrowType arrowType1 = new ArrowType(jSTypeRegistry0, node1, errorFunctionType0);
      boolean boolean0 = arrowType1.hasEqualParameters(arrowType0, false);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "20|-U,{3~\"SP");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      boolean boolean0 = arrowType0.checkArrowEquivalenceHelper(arrowType0, false);
      assertTrue(boolean0);
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "com.google.javascript.rhino.jstype.ArrowType");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) stack0);
      ArrowType arrowType1 = new ArrowType(jSTypeRegistry0, node0, errorFunctionType0);
      boolean boolean0 = arrowType0.checkArrowEquivalenceHelper(arrowType1, true);
      assertTrue(errorFunctionType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NumberType numberType0 = new NumberType(jSTypeRegistry0);
      JSType jSType0 = numberType0.autoboxesTo();
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, (Node) null, jSType0, true);
      arrowType0.hashCode();
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      VoidType voidType0 = new VoidType(jSTypeRegistry0);
      ImmutableList<JSType> immutableList0 = ImmutableList.of((JSType) voidType0, (JSType) voidType0, (JSType) voidType0, (JSType) voidType0, (JSType) voidType0);
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) immutableList0);
      Node node1 = new Node((-1), node0, node0, 49, 54);
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node1, voidType0, false);
      arrowType0.hashCode();
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "2%>fD&WxBp\"5xqj");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      arrowType0.resolveInternal(simpleErrorReporter0, noObjectType0);
      assertFalse(noObjectType0.isReturnTypeInferred());
      assertFalse(errorFunctionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "!)}3P");
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(errorFunctionType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[4];
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      ArrowType arrowType0 = new ArrowType(jSTypeRegistry0, node0, jSTypeArray0[0]);
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      JSType[] jSTypeArray0 = new JSType[7];
      JSTypeNative jSTypeNative0 = JSTypeNative.CHECKED_UNKNOWN_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      jSTypeArray0[0] = jSType0;
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node0);
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      ArrowType arrowType0 = errorFunctionType0.getInternalArrowType();
      boolean boolean0 = arrowType0.hasUnknownParamsOrReturn();
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "");
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, templateType0, (JSType) null);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType((Node) null, indexedType0);
      boolean boolean0 = arrowType0.hasAnyTemplateInternal();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoType noType0 = new NoType(jSTypeRegistry0);
      boolean boolean0 = noType0.hasAnyTemplate();
      assertFalse(boolean0);
      assertFalse(noType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      JSType[] jSTypeArray0 = new JSType[7];
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "YqW56 ");
      jSTypeArray0[1] = (JSType) templateType0;
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      ArrowType arrowType0 = jSTypeRegistry0.createArrowType(node0);
      boolean boolean0 = arrowType0.hasAnyTemplateInternal();
      assertTrue(boolean0);
  }
}
