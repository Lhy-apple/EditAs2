/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:16:43 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.FunctionInjector;
import com.google.javascript.jscomp.FunctionToBlockMutator;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.rhino.Node;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Set;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionInjector_ESTest extends FunctionInjector_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      Node node0 = Node.newString(3200, "$VALUES", (-688), 3200);
      Node node1 = new Node(1, node0, node0, node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, false);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, (Set<String>) null, functionInjector_InliningMode0, true, true);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      Node node0 = new Node((-688), (-688), (-688));
      Node node1 = new Node(1, node0, node0, node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, (Set<String>) null, functionInjector_InliningMode0, false, false);
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
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      Node node0 = Node.newString((-4603), "$VALUES", 61, 61);
      Node node1 = new Node(55, node0, node0, node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, true);
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, (Set<String>) null, functionInjector_InliningMode0, false, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      Node node0 = Node.newString((-4631), "$VALUES", (-679), (-679));
      Node node1 = new Node(47, node0, node0, 57, 56);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, true, false);
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node0, (Set<String>) null, functionInjector_InliningMode0, false, true);
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
      Node node0 = Node.newString((-319), "  parent:", (-319), 787);
      Node node1 = new Node(1, node0, node0);
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
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, true, true);
      JSModule jSModule0 = new JSModule("");
      Node node0 = new Node(52);
      ArrayList<FunctionInjector.Reference> arrayList0 = new ArrayList<FunctionInjector.Reference>();
      ImmutableBiMap<String, String> immutableBiMap0 = ImmutableBiMap.of("", "");
      ImmutableSet<String> immutableSet0 = immutableBiMap0.values();
      boolean boolean0 = functionInjector0.inliningLowersCost(jSModule0, node0, arrayList0, immutableSet0, true, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(comparator0).compare(any() , any());
      TreeSet<FunctionInjector.Reference> treeSet0 = new TreeSet<FunctionInjector.Reference>(comparator0);
      JSModule jSModule0 = new JSModule("com.google.javascript.jscomp.FunctionInjector$1");
      Node node0 = Node.newString((-2629), "com.google.javascript.jscomp.FunctionInjector$1", 748, 748);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, jSModule0, functionInjector_InliningMode0);
      treeSet0.add(functionInjector_Reference0);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, false, true);
      boolean boolean0 = functionInjector0.inliningLowersCost(jSModule0, node0, treeSet0, (Set<String>) null, true, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(supplier0);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, functionToBlockMutator_LabelNameSupplier0, true, true, true);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, (JSModule) null, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0);
      ArrayList<FunctionInjector.Reference> arrayList0 = new ArrayList<FunctionInjector.Reference>(immutableList0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost((JSModule) null, (Node) null, arrayList0, (Set<String>) null, true, false);
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
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      Node node0 = Node.newString(2246, "", 2246, (-56));
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(comparator0).compare(any() , any());
      TreeSet<FunctionInjector.Reference> treeSet0 = new TreeSet<FunctionInjector.Reference>(comparator0);
      JSModule jSModule0 = new JSModule("");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, jSModule0, functionInjector_InliningMode0);
      treeSet0.add(functionInjector_Reference0);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, node0, treeSet0, (Set<String>) null, false, false);
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
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(supplier0);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, functionToBlockMutator_LabelNameSupplier0, true, false, true);
      Node node0 = Node.newString(0, "O'n<%nFK\"P?", 0, 0);
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(comparator0).compare(any() , any());
      TreeSet<FunctionInjector.Reference> treeSet0 = new TreeSet<FunctionInjector.Reference>(comparator0);
      JSModule jSModule0 = new JSModule("/J7{&@wK{\"{=%NCQO");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, jSModule0, functionInjector_InliningMode0);
      treeSet0.add(functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, node0, treeSet0, (Set<String>) null, true, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, true, false);
      functionInjector0.setKnownConstants((Set<String>) null);
  }
}