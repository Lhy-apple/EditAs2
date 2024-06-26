/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:24:55 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.ConstCheck;
import com.google.javascript.jscomp.FunctionInjector;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.NameReferenceGraph;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PrepareAst;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.rhino.Node;
import java.util.ArrayList;
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
      CompilerOptions compilerOptions0 = new CompilerOptions();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      PrepareAst.PrepareAnnotations prepareAst_PrepareAnnotations0 = new PrepareAst.PrepareAnnotations();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, prepareAst_PrepareAnnotations0, (ScopeCreator) null);
      Node node0 = new Node(951);
      Node node1 = new Node(15, node0, node0, node0, 46, 46);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, compilerOptions0.aliasableStrings, functionInjector_InliningMode0, false, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.FunctionInjector", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      PrepareAst.PrepareAnnotations prepareAst_PrepareAnnotations0 = new PrepareAst.PrepareAnnotations();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, prepareAst_PrepareAnnotations0, (ScopeCreator) null);
      Node node0 = new Node(951);
      Node node1 = new Node(951, node0, node0, node0, 1, 103);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node0, compilerOptions0.stripNamePrefixes, functionInjector_InliningMode0, false, true);
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
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Node node0 = new Node((-1631));
      Node node1 = new Node(1, node0, node0, node0, 53, (-109));
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, false, true);
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, constCheck0);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      // Undeclared exception!
      try { 
        functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node1, compilerOptions0.stripTypes, functionInjector_InliningMode0, false, true);
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
      CompilerOptions compilerOptions0 = new CompilerOptions();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      PrepareAst.PrepareAnnotations prepareAst_PrepareAnnotations0 = new PrepareAst.PrepareAnnotations();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, prepareAst_PrepareAnnotations0, (ScopeCreator) null);
      Node node0 = new Node(951);
      Node node1 = new Node(15, node0, node0, node0, 46, 46);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      functionInjector0.canInlineReferenceToFunction(nodeTraversal0, node1, node0, compilerOptions0.stripNamePrefixes, functionInjector_InliningMode0, true, false);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      Node node0 = Node.newString("", 971, 971);
      Node node1 = new Node(971, node0, node0, node0, 1, 56);
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
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      ArrayList<FunctionInjector.Reference> arrayList0 = new ArrayList<FunctionInjector.Reference>();
      boolean boolean0 = functionInjector0.inliningLowersCost((JSModule) null, (Node) null, arrayList0, compilerOptions0.stripTypePrefixes, false, false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      Node node0 = new Node(951);
      JSModule jSModule0 = new JSModule((String) null);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, jSModule0, functionInjector_InliningMode0);
      ImmutableSetMultimap<FunctionInjector.Reference, FunctionInjector.Reference> immutableSetMultimap0 = ImmutableSetMultimap.of(functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0);
      ImmutableListMultimap<FunctionInjector.Reference, FunctionInjector.Reference> immutableListMultimap0 = ImmutableListMultimap.copyOf((Multimap<? extends FunctionInjector.Reference, ? extends FunctionInjector.Reference>) immutableSetMultimap0);
      ImmutableMultiset<FunctionInjector.Reference> immutableMultiset0 = immutableListMultimap0.keys();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, false, false);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, node0, immutableMultiset0, compilerOptions0.stripTypes, true, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, true, true);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "^HIyti6");
      JSModule jSModule0 = new JSModule("Unexpected call site type.");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, (JSModule) null, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule0, node0, immutableList0, (Set<String>) null, true, false);
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
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, false, true);
      JSModule jSModule0 = new JSModule("T");
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      JSModule jSModule1 = new JSModule("T");
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference((Node) null, jSModule0, functionInjector_InliningMode0);
      ImmutableList<FunctionInjector.Reference> immutableList0 = ImmutableList.of(functionInjector_Reference0, functionInjector_Reference0);
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost(jSModule1, (Node) null, immutableList0, (Set<String>) null, true, true);
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
      CompilerOptions compilerOptions0 = new CompilerOptions();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, false, false, false);
      Node node0 = new Node(951);
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.BLOCK;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(node0, (JSModule) null, functionInjector_InliningMode0);
      ImmutableSetMultimap<FunctionInjector.Reference, FunctionInjector.Reference> immutableSetMultimap0 = ImmutableSetMultimap.of(functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0);
      ImmutableListMultimap<FunctionInjector.Reference, FunctionInjector.Reference> immutableListMultimap0 = ImmutableListMultimap.copyOf((Multimap<? extends FunctionInjector.Reference, ? extends FunctionInjector.Reference>) immutableSetMultimap0);
      ImmutableMultiset<FunctionInjector.Reference> immutableMultiset0 = immutableListMultimap0.keys();
      // Undeclared exception!
      try { 
        functionInjector0.inliningLowersCost((JSModule) null, node0, immutableMultiset0, compilerOptions0.stripNameSuffixes, false, false);
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
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, false, true);
      Node node0 = compiler0.getJsRoot();
      NameReferenceGraph.Reference nameReferenceGraph_Reference0 = new NameReferenceGraph.Reference((Node) null, node0);
      JSModule jSModule0 = nameReferenceGraph_Reference0.getModule();
      FunctionInjector.InliningMode functionInjector_InliningMode0 = FunctionInjector.InliningMode.DIRECT;
      FunctionInjector.Reference functionInjector_Reference0 = new FunctionInjector.Reference(nameReferenceGraph_Reference0.parent, jSModule0, functionInjector_InliningMode0);
      ImmutableSetMultimap<FunctionInjector.Reference, FunctionInjector.Reference> immutableSetMultimap0 = ImmutableSetMultimap.of(functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0, functionInjector_Reference0);
      ImmutableListMultimap<FunctionInjector.Reference, FunctionInjector.Reference> immutableListMultimap0 = ImmutableListMultimap.copyOf((Multimap<? extends FunctionInjector.Reference, ? extends FunctionInjector.Reference>) immutableSetMultimap0);
      ImmutableMultiset<FunctionInjector.Reference> immutableMultiset0 = immutableListMultimap0.keys();
      boolean boolean0 = functionInjector0.inliningLowersCost(jSModule0, (Node) null, immutableMultiset0, (Set<String>) null, true, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      FunctionInjector functionInjector0 = new FunctionInjector(compiler0, supplier0, true, false, true);
      ImmutableBiMap<String, String> immutableBiMap0 = ImmutableBiMap.of("com.google.javascript.rhino.head.Node$PropListItem", "com.google.javascript.rhino.head.Node$PropListItem", "K8Dv XL/pt", "");
      ImmutableSet<String> immutableSet0 = immutableBiMap0.keySet();
      functionInjector0.setKnownConstants(immutableSet0);
      // Undeclared exception!
      try { 
        functionInjector0.setKnownConstants(immutableSet0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }
}
