/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:45:23 GMT 2023
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
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.EnumType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.testing.EmptyScope;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionTypeBuilder_ESTest extends FunctionTypeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("{", compiler0, node0, (String) null, scope0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.setSourceNode(node0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction((FunctionType) null, node0);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("|n");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("|n", compiler0, node0, "|n", scope0);
      Vector<JSType> vector0 = new Vector<JSType>();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) null, (List<JSType>) vector0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction(functionType0, (Node) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("", compiler0, node0, "", scope0);
      Vector<JSType> vector0 = new Vector<JSType>();
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) null, (List<JSType>) vector0);
      JSType[] jSTypeArray0 = new JSType[14];
      Node node1 = jSTypeRegistry0.createParameters(jSTypeArray0);
      functionTypeBuilder0.inferFromOverriddenFunction(functionType0, node1);
      functionTypeBuilder0.inferParameterTypes(node1, (JSDocInfo) null);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.FunctionTypeBuilder$1");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSTypeNative jSTypeNative0 = JSTypeNative.ARRAY_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction(functionType0, node0);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.FunctionTypeBuilder$1");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeNative jSTypeNative0 = JSTypeNative.DATE_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction(functionType0, node0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("{ne", compiler0, node0, "{ne", scope0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[7];
      Node node1 = jSTypeRegistry0.createParameters(jSTypeArray0);
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType(jSTypeArray0[5], false, jSTypeArray0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction(functionType0, node1);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("b#SVS&8", compiler0, node0, "Not declared as a constructor", scope0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferReturnType((JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.FunctionTypeBuilder$1");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferReturnType(jSDocInfo0);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferReturnStatementsAsLastResort((Node) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      // Undeclared exception!
      try { 
        functionTypeBuilder0.inferReturnStatementsAsLastResort(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionTypeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferInheritance((JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("<");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferInheritance(jSDocInfo0);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("<");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferThisType(jSDocInfo0, (JSType) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[1];
      EmptyScope emptyScope0 = new EmptyScope();
      JSType jSType0 = emptyScope0.getTypeOfThis();
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType(jSType0, false, jSTypeArray0);
      EnumType enumType0 = jSTypeRegistry0.createEnumType("JSC_EXTENDS_WITHOUT_TYPEDEF", functionType0);
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("b#SVS&8", compiler0, node0, "Not declared as a constructor", scope0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferThisType(jSDocInfo0, enumType0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{n");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("{n", compiler0, node0, (String) null, scope0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, (JSDocInfo) null);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      functionTypeBuilder1.inferThisType((JSDocInfo) null, functionType0);
      assertFalse(functionType0.isReturnTypeInferred());
      assertFalse(functionType0.hasCachedValues());
      assertFalse(functionType0.isConstructor());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "", (Scope) null);
      // Undeclared exception!
      try { 
        functionTypeBuilder0.inferThisType((JSDocInfo) null, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.JSTypeRegistry", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("b#SVS&8", compiler0, node0, "Not declared as a constructor", scope0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      // Undeclared exception!
      try { 
        functionTypeBuilder0.inferThisType(jSDocInfo0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.JSTypeRegistry", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferThisType(jSDocInfo0, (Node) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("|n");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.javascript.jscomp.mozilla.rhino.jdk13.VMBridge_jdk13", compiler0, node0, (String) null, scope0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes((Node) null, jSDocInfo0);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("xgA?Fr~U4q;_I' |");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("xgA?Fr~U4q;_I' |", compiler0, node0, "xgA?Fr~U4q;_I' |", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes((Node) null, (JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("", compiler0, node0, "", scope0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[14];
      Node node1 = jSTypeRegistry0.createParameters(jSTypeArray0);
      functionTypeBuilder0.inferParameterTypes(node1, (JSDocInfo) null);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[1];
      EmptyScope emptyScope0 = new EmptyScope();
      JSType jSType0 = emptyScope0.getTypeOfThis();
      Node node0 = jSTypeRegistry0.createParameters(jSTypeArray0);
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType(jSType0, false, jSTypeArray0);
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("b#SVS&8", compiler0, node0, "Not declared as a constructor", scope0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      functionTypeBuilder0.inferFromOverriddenFunction(functionType0, node0);
      functionTypeBuilder0.inferParameterTypes(node0, jSDocInfo0);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferTemplateTypeName((JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("|n");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("|n", compiler0, node0, (String) null, scope0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferTemplateTypeName(jSDocInfo0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("|n");
      Scope scope0 = new Scope(node0, compiler0);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("|n", compiler0, node0, (String) null, scope0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, jSDocInfo0);
      functionTypeBuilder1.buildAndRegister();
      FunctionType functionType0 = functionTypeBuilder1.buildAndRegister();
      assertFalse(functionType0.isReturnTypeInferred());
      assertFalse(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{ne");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
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
}