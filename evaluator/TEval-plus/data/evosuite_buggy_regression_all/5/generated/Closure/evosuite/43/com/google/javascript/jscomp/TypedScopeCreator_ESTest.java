/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:33:44 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{Q!y+i.vTl+Cd");
      Node node1 = new Node(37, node0, node0, node0, node0, 39, 45);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("TypedScopeCreator$DeferredSetTy4V", "TypedScopeCreator$DeferredSetTy4V");
      Node node1 = new Node(120, node0, node0, node0, 30, 30);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
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
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("qT:y<JYp");
      Node node1 = Node.newString(64, "qT:y<JYp");
      Node node2 = new Node(30, node1, node0, 53, 43);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node2, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("Q8.H");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node1 = compiler0.parseTestCode("Q8.H");
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("Q.H");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
      
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("uyQc");
      Node node1 = new Node(118, node0, node0, node0, 35, 12);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :35:12
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Node node0 = compiler0.parseTestCode("A=tB1S");
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{Q!y+i.vTl+Cd");
      Node node1 = new Node(86, node0, node0, node0, 4, 39);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("qyP:<JuOp");
      Node node1 = new Node(39, node0, node0, node0, 0, 2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("qP:<JuOp");
      Node node1 = new Node(41, node0, node0, 42, 4095);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("qT:y<JOp");
      Node node1 = new Node(43, node0, node0, node0, 38, 43);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("{Q!y+i.vTl+Cd");
      Node node1 = new Node(44, node0, node0, node0, 4, 39);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("M[", "M[");
      Node node1 = new Node(47, node0, node0, node0, node0, 12, 2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("TypedScopeCreator$DeferredSetType", "TypedScopeCreator$DeferredSetType");
      Node node1 = new Node(122, node0, node0, node0, 2163, 2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Node node0 = new Node(118);
      Node node1 = new Node(32, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, scope0);
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
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("Q8.H");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("Q8.H", "Q8.H");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.common.collect.ImmutableListMultimap");
      Node node1 = compiler0.parseTestCode("com.google.common.collect.ImmutableListMultimap");
      Node node2 = new Node(46, node0, node0, node1, 43, 2146);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node2, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }
}
