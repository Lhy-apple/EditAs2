/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:02:28 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerPass;
import com.google.javascript.jscomp.FunctionToBlockMutator;
import com.google.javascript.jscomp.MakeDeclaredNamesUnique;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.RenameLabels;
import com.google.javascript.rhino.Node;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MakeDeclaredNamesUnique_ESTest extends MakeDeclaredNamesUnique_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerPass compilerPass0 = MakeDeclaredNamesUnique.getContextualRenameInverter(compiler0);
      Node node0 = new Node((-1748302255));
      compilerPass0.process(node0, node0);
      assertEquals(1, Node.SPECIALCALL_EVAL);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("arguments", "arguments");
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
      assertEquals(39, Node.EMPTY_BLOCK);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      boolean boolean0 = makeDeclaredNamesUnique_ContextualRenamer0.stripConstIfReplaced();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer1 = (MakeDeclaredNamesUnique.ContextualRenamer)makeDeclaredNamesUnique_ContextualRenamer0.forChildScope();
      makeDeclaredNamesUnique_ContextualRenamer0.addDeclaredName("com.google.jasascript.jscomp.MakeDeclaredNadesUnique$ContextualRenameInvert+r");
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("com.google.jasascript.jscomp.MakeDeclaredNadesUnique$ContextualRenameInvert+r");
      assertFalse(makeDeclaredNamesUnique_ContextualRenamer1.equals((Object)makeDeclaredNamesUnique_ContextualRenamer0));
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "deterministic instanceof yields false", false);
      boolean boolean0 = makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "com.google.javascript.jscomp.MakeDeclaredNamesUnique$ContextualRenameInverter", true);
      makeDeclaredNamesUnique_InlineRenamer0.getReplacementName("com.google.javascript.jscomp.MakeDeclaredNamesUnique$ContextualRenameInverter");
      assertTrue(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "arguments", true);
      makeDeclaredNamesUnique_InlineRenamer0.forChildScope();
      assertTrue(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String string0 = MakeDeclaredNamesUnique.ContextualRenameInverter.getOrginalName("$$");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(renameLabels_DefaultNameSupplier0);
      MakeDeclaredNamesUnique.BoilerplateRenamer makeDeclaredNamesUnique_BoilerplateRenamer0 = new MakeDeclaredNamesUnique.BoilerplateRenamer(functionToBlockMutator_LabelNameSupplier0, "");
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique_BoilerplateRenamer0.forChildScope();
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
      Node node0 = new Node(105);
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique((MakeDeclaredNamesUnique.Renamer) null);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105, 105, 105);
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
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
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105, 105, 105);
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique0);
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique0.shouldTraverse(nodeTraversal0, node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MakeDeclaredNamesUnique", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((double) 120);
      Node node1 = new Node(120, node0, node0, node0, node0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node1, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Node node0 = new Node(105);
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique((MakeDeclaredNamesUnique.Renamer) null);
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique0.visit((NodeTraversal) null, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MakeDeclaredNamesUnique.ContextualRenameInverter makeDeclaredNamesUnique_ContextualRenameInverter0 = (MakeDeclaredNamesUnique.ContextualRenameInverter)MakeDeclaredNamesUnique.getContextualRenameInverter(compiler0);
      Node node0 = new Node(105);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique_ContextualRenameInverter0);
      Node node1 = new Node(120, node0, node0, node0, node0);
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique(makeDeclaredNamesUnique_ContextualRenamer0);
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique0.visit(nodeTraversal0, node1, node1);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      Node node1 = compiler0.parseTestCode("some");
      node1.addChildToFront(node0);
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node1, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = MakeDeclaredNamesUnique.ContextualRenameInverter.getOrginalName("arguments");
      assertEquals("arguments", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(renameLabels_DefaultNameSupplier0);
      MakeDeclaredNamesUnique.BoilerplateRenamer makeDeclaredNamesUnique_BoilerplateRenamer0 = new MakeDeclaredNamesUnique.BoilerplateRenamer(functionToBlockMutator_LabelNameSupplier0, "arguments");
      makeDeclaredNamesUnique_BoilerplateRenamer0.addDeclaredName("arguments");
      assertFalse(makeDeclaredNamesUnique_BoilerplateRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer1 = (MakeDeclaredNamesUnique.ContextualRenamer)makeDeclaredNamesUnique_ContextualRenamer0.forChildScope();
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("]2`yc*-?Z6WXFL");
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("]2`yc*-?Z6WXFL");
      assertFalse(makeDeclaredNamesUnique_ContextualRenamer1.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(renameLabels_DefaultNameSupplier0);
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(functionToBlockMutator_LabelNameSupplier0, "arguments", true);
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("arguments");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, ",", true);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("");
      assertTrue(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(renameLabels_DefaultNameSupplier0);
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(functionToBlockMutator_LabelNameSupplier0, "Using the debugger statement can halt your application if the user has a JavaScript debugger running.", true);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("Using the debugger statement can halt your application if the user has a JavaScript debugger running.");
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("Using the debugger statement can halt your application if the user has a JavaScript debugger running.");
      assertTrue(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      FunctionToBlockMutator.LabelNameSupplier functionToBlockMutator_LabelNameSupplier0 = new FunctionToBlockMutator.LabelNameSupplier(renameLabels_DefaultNameSupplier0);
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(functionToBlockMutator_LabelNameSupplier0, "Using the debugger statement can halt your application if the user has a JavaScript debugger running.", true);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("$$");
      assertTrue(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }
}
