/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:06:50 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SyntacticScopeCreator_ESTest extends SyntacticScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0, (SyntacticScopeCreator.RedeclarationHandler) null);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("non-monoInic data-floz analysis", "non-monoInic data-floz analysis");
      Node node1 = new Node(105, node0, node0, node0);
      // Undeclared exception!
      try { 
        syntacticScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.SyntacticScopeCreator", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("non-mon~Inic data-floz ana3ysis", "non-mon~Inic data-floz ana3ysis");
      Node node1 = new Node((-999), node0, node0, node0);
      Scope scope0 = syntacticScopeCreator0.createScope(node1, (Scope) null);
      // Undeclared exception!
      try { 
        syntacticScopeCreator0.createScope(node0, scope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("non-monoInic data-floz analysis", "non-monoInic data-floz analysis");
      Node node1 = new Node(118, node0, node0, node0);
      // Undeclared exception!
      try { 
        syntacticScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("non-mon~Inic data-floz ana3ysis", "non-mon~Inic data-floz ana3ysis");
      Node node1 = new Node(120, node0, node0, node0);
      // Undeclared exception!
      try { 
        syntacticScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("{}", "{}");
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Scope scope0 = syntacticScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("non-monotonic data-flow analysis", "");
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = new Node(118, (-3), 12);
      Scope scope0 = syntacticScopeCreator0.createScope(node0, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("non-mon,tnic data-flow analsis", "non-mon,tnic data-flow analsis");
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = new Node(114, 16, 8);
      Scope scope0 = syntacticScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }
}