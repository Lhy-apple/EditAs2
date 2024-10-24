/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:41:16 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConstCheck;
import com.google.javascript.jscomp.FunctionInjector;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.rhino.Node;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;
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
      JSModule jSModule0 = new JSModule("");
      ArrayList<FunctionInjector.Reference> arrayList0 = new ArrayList<FunctionInjector.Reference>();
      boolean boolean0 = functionInjector0.inliningLowersCost(jSModule0, (Node) null, arrayList0, (Set<String>) null, true, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = new Node(83);
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, constCheck0);
      Node node1 = new Node(42, node0, node0, node0);
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, treeSet0, functionInjector_InliningMode0, false, false);
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
      Node node0 = new Node(83);
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, true, true);
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, constCheck0);
      Node node1 = new Node(48, node0, node0, node0);
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, treeSet0, functionInjector_InliningMode0, false, true);
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
      Node node0 = new Node(83);
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, constCheck0);
      Node node1 = new Node(48, node0, node0, node0);
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, treeSet0, functionInjector_InliningMode0, false, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Node node0 = new Node(83);
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, constCheck0);
      Node node1 = new Node(4095, node0, node0, node0);
      functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, treeSet0, functionInjector_InliningMode0, true, true);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = new Node(83);
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      Node node1 = new Node(42, node0, node0, node0);
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
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, (JSModule) null, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost((JSModule) null, (Node) null, immutableList0, treeSet0, true, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = new Node(83);
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      TreeSet<String> treeSet0 = new TreeSet<String>();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, true);
      JSModule jSModule0 = new JSModule("");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, jSModule0, functionInjector_InliningMode0);
      FunctionInjector.Reference functionInjector_Reference1 = new FunctionInjector.Reference(node0, (JSModule) null, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference1, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference1, functionInjector_Reference1, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference1, functionInjector_Reference0, functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, node0, immutableList0, treeSet0, true, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, true);
      JSModule jSModule0 = new JSModule((String) null);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule0, functionInjector_InliningMode0);
      JSModule jSModule1 = new JSModule((String) null);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule1, (Node) null, immutableList0, treeSet0, true, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionInjector", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      JSModule jSModule0 = new JSModule("com.google.javascript.jscomp.NameReferenceGraphConstruction$Traversal");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule0, functionInjector_InliningMode0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(treeSet0);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, false, false);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, (Node) null, immutableList0, linkedHashSet0, false, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      JSModule jSModule0 = new JSModule("");
      ArrayList<FunctionInjector.Reference> arrayList0 = new ArrayList<FunctionInjector.Reference>();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule0, functionInjector_InliningMode0);
      arrayList0.add(functionInjector_Reference0);
      boolean boolean0 = functionInjector0.inliningLowersCost(jSModule0, (Node) null, arrayList0, (Set<String>) null, true, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      JSModule jSModule0 = new JSModule("");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule0, functionInjector_InliningMode0);
      TreeSet<String> treeSet0 = new TreeSet<String>();
      LinkedList<Locale.LanguageRange> linkedList0 = new LinkedList<Locale.LanguageRange>();
      List<String> list0 = Locale.filterTags((List<Locale.LanguageRange>) linkedList0, (Collection<String>) treeSet0);
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(list0);
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, true);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, (Node) null, immutableList0, linkedHashSet0, true, true);
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
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      functionInjector0.setKnownConstants((Set<String>) null);
  }
}
