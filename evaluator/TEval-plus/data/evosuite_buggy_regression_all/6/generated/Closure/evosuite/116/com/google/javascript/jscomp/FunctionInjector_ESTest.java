/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:02:40 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.FindExportableNodes;
import com.google.javascript.jscomp.FunctionInjector;
import com.google.javascript.jscomp.FunctionToBlockMutator;
import com.google.javascript.jscomp.InlineSimpleMethods;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.NameAnalyzer;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.rhino.Node;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionInjector_ESTest extends FunctionInjector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      FindExportableNodes findExportableNodes0 = new FindExportableNodes(compiler0, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, findExportableNodes0, (ScopeCreator) null);
      Node node0 = Node.newNumber(4291.05412778093);
      Node node1 = new Node(30, node0, node0, 15, 55);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node0, linkedHashSet0, functionInjector_InliningMode0, false, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      Node node0 = Node.newString(2, "#*+rKLSN;u-!", 2, 2);
      Node node1 = new Node(2, node0, node0, 4, 54);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction((NodeTraversal) null, node1, node1, linkedHashSet0, functionInjector_InliningMode0, false, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionInjector", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, true);
      Node node0 = Node.newString(2, "#*+rKLSN;u-!", 2, 2);
      Node node1 = new Node(2, node0, node0, 4, 54);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      functionInjector0.canInlineReferenceToFunction((NodeTraversal) null, node1, node1, linkedHashSet0, functionInjector_InliningMode0, true, true);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      // Undeclared exception!
      try { 
        functionInjector0.inline((Node) null, "g^`(ZH[!|R_YkI", (Node) null, functionInjector_InliningMode0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionInjector", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      // Undeclared exception!
      try { 
        functionInjector0.inline((Node) null, "EXPRESSION", (Node) null, functionInjector_InliningMode0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionInjector", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, true);
      Node node0 = Node.newNumber(4291.05412778093);
      Node node1 = new Node(0, node0, node0, 15, 8);
      // Undeclared exception!
      try { 
        functionInjector0.maybePrepareCall(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, true);
      InlineSimpleMethods inlineSimpleMethods0 = new InlineSimpleMethods(compiler0);
      Set<String> set0 = inlineSimpleMethods0.nonMethodProperties;
      JSModule jSModule0 = new JSModule("com.google.javascript.jscomp.ReferenceCollectingCallback$BasicBlock");
      HashSet<FunctionInjector.Reference> hashSet0 = new HashSet<FunctionInjector.Reference>();
      boolean boolean0 = functionInjector0.inliningLowersCost(jSModule0, (Node) null, hashSet0, set0, true, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, true);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, (JSModule) null, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0);
      InlineSimpleMethods inlineSimpleMethods0 = new InlineSimpleMethods(compiler0);
      Set<String> set0 = inlineSimpleMethods0.externMethods;
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost((JSModule) null, (Node) null, immutableList0, set0, true, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier((Supplier<String>) null);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, functionToBlockMutator_LabelNameSupplier0, true, true, true);
      JSModule jSModule0 = new JSModule("");
      Node node0 = Node.newString(0, "f*;^Iio^szi lfx,", 2003, 620);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, (JSModule) null, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, node0, immutableList0, (Set<String>) null, true, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      JSModule jSModule0 = new JSModule("");
      JSModule jSModule1 = new JSModule((String) null);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule1, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, (Node) null, immutableList0, (Set<String>) null, true, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionInjector", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      JSModule jSModule0 = new JSModule("H?6MK-59");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule0, functionInjector_InliningMode0);
      FunctionInjector.Reference[] functionInjector_ReferenceArray0 = new FunctionInjector.Reference[1];
      functionInjector_ReferenceArray0[0] = functionInjector_Reference0;
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.copyOf(functionInjector_ReferenceArray0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, (Node) null, immutableList0, (Set<String>) null, true, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      JSModule jSModule0 = new JSModule("H?6MK-59");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule0, functionInjector_InliningMode0);
      FunctionInjector.Reference[] functionInjector_ReferenceArray0 = new FunctionInjector.Reference[1];
      functionInjector_ReferenceArray0[0] = functionInjector_Reference0;
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.copyOf(functionInjector_ReferenceArray0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, (Node) null, immutableList0, (Set<String>) null, false, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, true);
      Set<String> set0 = NameAnalyzer.DEFAULT_GLOBAL_NAMES;
      functionInjector0.setKnownConstants(set0);
      // Undeclared exception!
      try { 
        functionInjector0.setKnownConstants(set0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }
}