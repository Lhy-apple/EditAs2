/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:36:35 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.FunctionTypeBuilder;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NoType;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionTypeBuilder_ESTest extends FunctionTypeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("dW+", "dW+");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.common.collect.AbstractMapBasedMultiset", compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      NoType noType0 = (NoType)functionType0.getRestrictedTypeGivenToBooleanOutcome(false);
      JSType[] jSTypeArray0 = new JSType[5];
      jSTypeArray0[0] = (JSType) functionType0;
      jSTypeArray0[1] = (JSType) functionType0;
      jSTypeArray0[2] = (JSType) noType0;
      jSTypeArray0[3] = (JSType) noType0;
      jSTypeArray0[4] = (JSType) functionType0;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction(noType0, (Node) null);
      functionTypeBuilder1.inferParameterTypes(node1, jSDocInfo0);
      assertEquals(1, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("i=4uIUO-LR%Y:+W\b");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("i=4uIUO-LR%Y:+W\b", compiler0, node0, "i=4uIUO-LR%Y:+W\b", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionTypeBuilder.AstFunctionContents functionTypeBuilder_AstFunctionContents0 = new FunctionTypeBuilder.AstFunctionContents(node0);
      FunctionTypeBuilder functionTypeBuilder2 = functionTypeBuilder1.setContents(functionTypeBuilder_AstFunctionContents0);
      // Undeclared exception!
      try { 
        functionTypeBuilder2.buildAndRegister();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      FunctionTypeBuilder.AstFunctionContents functionTypeBuilder_AstFunctionContents0 = new FunctionTypeBuilder.AstFunctionContents((Node) null);
      assertFalse(functionTypeBuilder_AstFunctionContents0.mayHaveNonEmptyReturns());
      
      functionTypeBuilder_AstFunctionContents0.recordNonEmptyReturn();
      assertTrue(functionTypeBuilder_AstFunctionContents0.mayHaveNonEmptyReturns());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      FunctionTypeBuilder.UnknownFunctionContents functionTypeBuilder_UnknownFunctionContents0 = new FunctionTypeBuilder.UnknownFunctionContents();
      Iterable<String> iterable0 = functionTypeBuilder_UnknownFunctionContents0.getEscapedVarNames();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      FunctionTypeBuilder.UnknownFunctionContents functionTypeBuilder_UnknownFunctionContents0 = new FunctionTypeBuilder.UnknownFunctionContents();
      boolean boolean0 = functionTypeBuilder_UnknownFunctionContents0.mayBeFromExterns();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("hm");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "hm", (Scope) null);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.setContents((FunctionTypeBuilder.FunctionContents) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.koogle.common.io.CharS5reams");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.koogle.common.io.CharS5reams", compiler0, node0, "com.koogle.common.io.CharS5reams", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction((FunctionType) null, node0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("dW+", "dW+");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.common.collect.AbstractMapBasedMultiset", compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType jSType0 = functionType0.getRestrictedTypeGivenToBooleanOutcome(false);
      JSType[] jSTypeArray0 = new JSType[5];
      jSTypeArray0[0] = (JSType) functionType0;
      jSTypeArray0[1] = (JSType) functionType0;
      jSTypeArray0[2] = jSType0;
      jSTypeArray0[3] = jSType0;
      jSTypeArray0[4] = (JSType) functionType0;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      functionTypeBuilder0.inferFromOverriddenFunction(functionType0, node1);
      assertEquals(0, compiler0.getWarningCount());
      assertFalse(functionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("dW+", "dW+");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.common.collect.AbstractMapBasedMultiset", compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      NoType noType0 = (NoType)functionType0.getRestrictedTypeGivenToBooleanOutcome(false);
      JSType[] jSTypeArray0 = new JSType[5];
      jSTypeArray0[0] = (JSType) functionType0;
      jSTypeArray0[1] = (JSType) functionType0;
      jSTypeArray0[2] = (JSType) noType0;
      jSTypeArray0[3] = (JSType) noType0;
      jSTypeArray0[4] = (JSType) functionType0;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      functionTypeBuilder0.inferFromOverriddenFunction(noType0, node1);
      assertEquals(0, compiler0.getWarningCount());
      assertFalse(functionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("dW+", "dW+");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.common.collect.AbstractMapBasedMultiset", compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType jSType0 = functionType0.getRestrictedTypeGivenToBooleanOutcome(false);
      assertFalse(functionType0.isReturnTypeInferred());
      
      JSType[] jSTypeArray0 = new JSType[5];
      jSTypeArray0[0] = (JSType) functionType0;
      jSTypeArray0[1] = (JSType) functionType0;
      jSTypeArray0[2] = jSType0;
      jSTypeArray0[3] = jSType0;
      jSTypeArray0[4] = (JSType) functionType0;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      JSTypeNative jSTypeNative0 = JSTypeNative.SYNTAX_ERROR_FUNCTION_TYPE;
      FunctionType functionType1 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      functionTypeBuilder0.inferFromOverriddenFunction(functionType1, node1);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("UhELl", "UhELl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhELl", compiler0, node0, "UhELl", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      FunctionType functionType1 = jSTypeRegistry0.createFunctionType((JSType) functionType0, node0);
      FunctionTypeBuilder functionTypeBuilder2 = functionTypeBuilder1.inferFromOverriddenFunction(functionType1, node0);
      JSType[] jSTypeArray0 = new JSType[9];
      jSTypeArray0[0] = (JSType) functionType1;
      jSTypeArray0[1] = (JSType) functionType0;
      jSTypeArray0[2] = (JSType) functionType0;
      jSTypeArray0[3] = (JSType) functionType0;
      jSTypeArray0[4] = (JSType) functionType1;
      jSTypeArray0[5] = (JSType) functionType0;
      jSTypeArray0[6] = (JSType) functionType0;
      jSTypeArray0[7] = (JSType) functionType1;
      jSTypeArray0[8] = (JSType) functionType1;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      functionTypeBuilder2.inferParameterTypes(node1, jSDocInfo0);
      assertEquals(0, compiler0.getWarningCount());
      assertFalse(functionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferReturnType((JSDocInfo) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("UhLl", "UhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhLl", compiler0, node0, "UhLl", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferReturnType(jSDocInfo0);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("h");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("h", compiler0, node0, "h", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferInheritance((JSDocInfo) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferInheritance(jSDocInfo0);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("NPapr0arC2");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("NPapr0arC2", compiler0, node0, "NPapr0arC2", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder1.buildAndRegister();
      functionTypeBuilder0.inferThisType(jSDocInfo0, functionType0);
      functionTypeBuilder0.inferThisType(jSDocInfo0, functionType0);
      assertFalse(functionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("d<:MSoAF2ba$Iw^9");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("d<:MSoAF2ba$Iw^9", compiler0, node0, "d<:MSoAF2ba$Iw^9", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferThisType(jSDocInfo0, (JSType) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("hA9", "hA9");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("hA9", compiler0, node0, "hA9", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, (JSDocInfo) null);
      FunctionType functionType0 = functionTypeBuilder1.buildAndRegister();
      functionTypeBuilder1.inferThisType((JSDocInfo) null, functionType0);
      assertFalse(functionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes((Node) null, jSDocInfo0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes((Node) null, (JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("dW+");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("dW+", compiler0, node0, "dW+", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, (JSDocInfo) null);
      FunctionType functionType0 = functionTypeBuilder1.buildAndRegister();
      assertFalse(functionType0.isReturnTypeInferred());
      
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[1];
      jSTypeArray0[0] = (JSType) functionType0;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      functionTypeBuilder1.inferParameterTypes(node1, (JSDocInfo) null);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferTemplateTypeName((JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferTemplateTypeName(jSDocInfo0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("dW+", "dW+");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.common.collect.AbstractMapBasedMultiset", compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType jSType0 = functionType0.getRestrictedTypeGivenToBooleanOutcome(false);
      JSType[] jSTypeArray0 = new JSType[5];
      jSTypeArray0[0] = (JSType) functionType0;
      jSTypeArray0[1] = (JSType) functionType0;
      jSTypeArray0[2] = jSType0;
      jSTypeArray0[3] = jSType0;
      jSTypeArray0[4] = (JSType) functionType0;
      Node node1 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      JSTypeNative jSTypeNative0 = JSTypeNative.SYNTAX_ERROR_FUNCTION_TYPE;
      FunctionType functionType1 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction(functionType1, (Node) null);
      functionTypeBuilder1.inferParameterTypes(node1, jSDocInfo0);
      assertEquals(1, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("UhhLl", "UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(jSDocInfo0);
      functionTypeBuilder1.buildAndRegister();
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      assertFalse(functionType0.isReturnTypeInferred());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("UhhLl");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("UhhLl", compiler0, node0, "UhhLl", (Scope) null);
      // Undeclared exception!
      try { 
        functionTypeBuilder0.buildAndRegister();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // All Function types must have params and a return type
         //
         verifyException("com.google.javascript.jscomp.FunctionTypeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      boolean boolean0 = FunctionTypeBuilder.isFunctionTypeDeclaration(jSDocInfo0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      FunctionTypeBuilder.AstFunctionContents functionTypeBuilder_AstFunctionContents0 = new FunctionTypeBuilder.AstFunctionContents((Node) null);
      functionTypeBuilder_AstFunctionContents0.recordEscapedVarName("goog.isDef");
      Iterable<String> iterable0 = functionTypeBuilder_AstFunctionContents0.getEscapedVarNames();
      assertTrue(iterable0.contains("goog.isDef"));
      assertFalse(functionTypeBuilder_AstFunctionContents0.mayHaveNonEmptyReturns());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      FunctionTypeBuilder.AstFunctionContents functionTypeBuilder_AstFunctionContents0 = new FunctionTypeBuilder.AstFunctionContents((Node) null);
      Iterable<String> iterable0 = functionTypeBuilder_AstFunctionContents0.getEscapedVarNames();
      assertFalse(functionTypeBuilder_AstFunctionContents0.mayHaveNonEmptyReturns());
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      FunctionTypeBuilder.AstFunctionContents functionTypeBuilder_AstFunctionContents0 = new FunctionTypeBuilder.AstFunctionContents((Node) null);
      functionTypeBuilder_AstFunctionContents0.recordEscapedVarName(" 4n3EK~9");
      functionTypeBuilder_AstFunctionContents0.recordEscapedVarName(" 4n3EK~9");
      assertFalse(functionTypeBuilder_AstFunctionContents0.mayHaveNonEmptyReturns());
  }
}