/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:33:57 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("on");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(120, node0, node0, node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(CATCH):  [testcode] :-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_returnArg(JSCompiler_returnArg_value) {  return function() {return JSCompiler_returnArg_value}}");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node1 = new Node(32, node0, node0, node0, node0, 5, 4);
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("avaulabN");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(118, node0, node0, node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("qQ=kquTT");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscmp");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(41, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("1");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(43);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("msg.X.ot.available");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(44);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(47, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("ImmutableMultiset$EntrySet");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Node node0 = compiler0.parseTestCode("j");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(64, node0);
      typedScopeCreator0.createScope(node1, (Scope) null);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("jcmp");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(69, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("msbad.jsctag");
      Node node1 = new Node(122, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("1");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(37, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("avaulablN");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(118);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("avaulablN", "avaulablN");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node1 = new Node(118);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node1.setJSDocInfo(jSDocInfo0);
      Node node2 = new Node(2, node0, node0, node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_returnArg(JSCompiler_returnArg_value) {  return function() {return JSCompiler_returnArg_value}}");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("avaulablN");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(86);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node1.setJSDocInfo(jSDocInfo0);
      Node node2 = new Node(2101, node0, node0, node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, (Scope) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 2101
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("com.google.javascript.jscmp");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscmp", "com.google.javascript.jscmp");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("goog.typedef");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("43%=%=75lnC=u");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[0];
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs(jSTypeArray0);
      Node node2 = new Node(34, node1, node1, node0);
      Scope scope0 = new Scope(node1, compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}