/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:38:26 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("JSC_ENUM_INITIALmZER_NOT_PNUV");
      Node node1 = new Node(37, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(120, 27, 43);
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
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.mozilla.rhino.ast.BreakStatement");
      Node node1 = new Node(47, node0, node0, 52, 3);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("JSC_ENUv_INITIALmZER_NOT_PNUV", "JSC_ENUv_INITIALmZER_NOT_PNUV");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(118, node0, node0, node0, 19, 16);
      // Undeclared exception!
      try { 
        typedScopeCreator0.patchGlobalScope((Scope) null, node1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.mozilla.rhino.ast.BreakStatement");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
      
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("var t;");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(" te=tcode ");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("JSC_ENUv_INITIALmZER_NOT_PNUV", "JSC_ENUv_INITIALmZER_NOT_PNUV");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(118, node0, node0, node0, 19, 16);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(105, 105, 40);
      Node node1 = new Node(31, node0, node0, node0, 30, 40);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
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
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("0x.$3");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("JSC_ENUv_INITIALmZER_NOT_PNUV");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(41, node0, node0, node0, 3, 27);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("JSC_LENDS_ON_NON_OBJECT");
      Node node1 = new Node(43, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("JSC_ENUv_INITIALmZER_NOT_PNUV");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(44, node0, node0, node0, 33, 14);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(47, 82, 47);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(69, 69, 69);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("M']OscE}Jy");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(122, node0, node0, node0, 2758, 4095);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("var t;");
      Node node1 = new Node(429, 25, 50);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      compiler0.parseTestCode("var wH");
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.mozilla.rhino.ast.BreakStatement", "com.google.javascript.jscomp.mozilla.rhino.ast.BreakStatement");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.mozilla.rhino.ast.BreakStatement");
      Node node1 = compiler0.parseTestCode("com.google.javascript.jscomp.mozilla.rhino.ast.BreakStatement");
      Node node2 = new Node((-1), node0, node1, node1, 52, 53);
      Scope scope0 = typedScopeCreator0.createScope(node2, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      Vector<JSType> vector0 = new Vector<JSType>();
      Node node0 = jSTypeRegistry0.createParameters((List<JSType>) vector0);
      Node node1 = new Node(2, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}