/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:38:02 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.FunctionTypeBuilder;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.RecordType;
import java.util.HashMap;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionTypeBuilder_ESTest extends FunctionTypeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "2", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.setSourceNode(node0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes((Node) null, jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      functionTypeBuilder1.inferFromOverriddenFunction(functionType0, (Node) null);
      assertFalse(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      Node node1 = new Node(37, node0, node0, 2, 13);
      // Undeclared exception!
      try { 
        functionTypeBuilder1.inferFromOverriddenFunction(functionType0, node1);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // SCRIPT 1 [sourcename: java.lang.String@0000000485] [synthetic: 1] is not a string node
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      Node node1 = new Node(1573, node0, node0, 19, 92);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      HashMap<String, JSType> hashMap0 = new HashMap<String, JSType>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      ObjectType objectType0 = recordType0.getImplicitPrototype();
      JSType[] jSTypeArray0 = new JSType[9];
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) objectType0, false, jSTypeArray0);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferFromOverriddenFunction(functionType0, node1);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("^", "^");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.javascript.jscomp.NameAnalyzer$NameInformation", compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferReturnType((JSDocInfo) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferReturnType(jSDocInfo0);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferInheritance((JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "2", (Scope) null);
      functionTypeBuilder0.inferInheritance(jSDocInfo0);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferThisType(jSDocInfo0, (JSType) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "2", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder1.buildAndRegister();
      functionTypeBuilder0.inferThisType(jSDocInfo0, functionType0);
      assertFalse(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes((Node) null, jSDocInfo0);
      FunctionType functionType0 = functionTypeBuilder0.buildAndRegister();
      functionTypeBuilder1.inferThisType((JSDocInfo) null, functionType0);
      assertFalse(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
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
  public void test12()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(".", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("&D|f)6;#Tk<f'.'", compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferThisType(jSDocInfo0, (Node) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
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
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes((Node) null, (JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("^", "^");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("com.google.javascript.jscomp.NameAnalyzer$NameInformation", compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, (JSDocInfo) null);
      assertSame(functionTypeBuilder1, functionTypeBuilder0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      Node node1 = Node.newString(4, "", 0, 29);
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "2", (Scope) null);
      Node node2 = new Node(523, node1, node0, 17, (-1629));
      // Undeclared exception!
      try { 
        functionTypeBuilder0.inferParameterTypes(node2, jSDocInfo0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // SCRIPT 1 [sourcename: java.lang.String@0000000483] [synthetic: 1] is not a string node
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder("", compiler0, node0, "optional arguments must be at the end", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferTemplateTypeName((JSDocInfo) null);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, (String) null, (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferTemplateTypeName(jSDocInfo0);
      assertSame(functionTypeBuilder0, functionTypeBuilder1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
      FunctionTypeBuilder functionTypeBuilder0 = new FunctionTypeBuilder((String) null, compiler0, node0, "2", (Scope) null);
      FunctionTypeBuilder functionTypeBuilder1 = functionTypeBuilder0.inferParameterTypes(node0, jSDocInfo0);
      functionTypeBuilder1.buildAndRegister();
      FunctionType functionType0 = functionTypeBuilder1.buildAndRegister();
      assertFalse(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("", "");
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
  public void test21()  throws Throwable  {
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      boolean boolean0 = FunctionTypeBuilder.isFunctionTypeDeclaration(jSDocInfo0);
      assertFalse(boolean0);
  }
}
