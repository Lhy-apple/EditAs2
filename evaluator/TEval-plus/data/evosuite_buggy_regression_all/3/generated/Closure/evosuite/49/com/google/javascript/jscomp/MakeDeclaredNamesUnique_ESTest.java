/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:11:36 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerPass;
import com.google.javascript.jscomp.MakeDeclaredNamesUnique;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
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
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Node node0 = new Node((-54));
      NodeTraversal.traverse((AbstractCompiler) null, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
      NodeTraversal.traverse((AbstractCompiler) null, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.BoilerplateRenamer makeDeclaredNamesUnique_BoilerplateRenamer0 = new MakeDeclaredNamesUnique.BoilerplateRenamer(renameLabels_DefaultNameSupplier0, ";@8O=cR}cvN)9/ =");
      boolean boolean0 = makeDeclaredNamesUnique_BoilerplateRenamer0.stripConstIfReplaced();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "arguments", "arguments");
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer1 = (MakeDeclaredNamesUnique.ContextualRenamer)makeDeclaredNamesUnique_ContextualRenamer0.forChildScope();
      makeDeclaredNamesUnique_ContextualRenamer0.addDeclaredName("@`>G)=a");
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("@`>G)=a");
      assertFalse(makeDeclaredNamesUnique_ContextualRenamer1.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "arguments", false);
      boolean boolean0 = makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer((Supplier<String>) null, "@`>G)=a", false);
      makeDeclaredNamesUnique_InlineRenamer0.getReplacementName("@`>G)=a");
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "arguments", true);
      makeDeclaredNamesUnique_InlineRenamer0.forChildScope();
      assertTrue(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerPass compilerPass0 = MakeDeclaredNamesUnique.getContextualRenameInverter(compiler0);
      // Undeclared exception!
      try { 
        compiler0.process(compilerPass0);
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
  public void test08()  throws Throwable  {
      String string0 = MakeDeclaredNamesUnique.ContextualRenameInverter.getOrginalName("$$");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MakeDeclaredNamesUnique.ContextualRenameInverter makeDeclaredNamesUnique_ContextualRenameInverter0 = (MakeDeclaredNamesUnique.ContextualRenameInverter)MakeDeclaredNamesUnique.getContextualRenameInverter(compiler0);
      Node node0 = Node.newNumber(0.0);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique_ContextualRenameInverter0);
      assertEquals(19, Node.LABEL_PROP);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Node node0 = new Node(105, (-2641), (-2641));
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.BoilerplateRenamer makeDeclaredNamesUnique_BoilerplateRenamer0 = new MakeDeclaredNamesUnique.BoilerplateRenamer(renameLabels_DefaultNameSupplier0, "<html><body><style type=\"text/css\">body, td, p {font-family: Arial; font-size: 83%} ul {margin-top:2px; margin-left:0px; padding-left:1em;} li {margin-top:3px; margin-left:24px; padding-left:0px;padding-bottom: 4px}</style>");
      MakeDeclaredNamesUnique.Renamer makeDeclaredNamesUnique_Renamer0 = makeDeclaredNamesUnique_BoilerplateRenamer0.forChildScope();
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique(makeDeclaredNamesUnique_Renamer0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) null, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105, 105, 105);
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
  public void test12()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Node node0 = new Node(105, 105, 105);
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique0.shouldTraverse((NodeTraversal) null, node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MakeDeclaredNamesUnique", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(120, "arguments", 120, 120);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Node node0 = new Node(105);
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
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MakeDeclaredNamesUnique.ContextualRenameInverter makeDeclaredNamesUnique_ContextualRenameInverter0 = (MakeDeclaredNamesUnique.ContextualRenameInverter)MakeDeclaredNamesUnique.getContextualRenameInverter(compiler0);
      Node node0 = new Node(120);
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique_ContextualRenameInverter0);
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = MakeDeclaredNamesUnique.ContextualRenameInverter.getOrginalName("arguments");
      assertEquals("arguments", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      makeDeclaredNamesUnique_ContextualRenamer0.addDeclaredName("arguments");
      assertFalse(makeDeclaredNamesUnique_ContextualRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer1 = (MakeDeclaredNamesUnique.ContextualRenamer)makeDeclaredNamesUnique_ContextualRenamer0.forChildScope();
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("@`>G)=a");
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("@`>G)=a");
      assertFalse(makeDeclaredNamesUnique_ContextualRenamer1.equals((Object)makeDeclaredNamesUnique_ContextualRenamer0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = null;
      try {
        makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "", true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "arguments", false);
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
  public void test21()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "arguments", false);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("$$");
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "com.google.javascript.jscomp.MakeDeclaredNamesUnique", false);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("");
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("");
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      RenameLabels.DefaultNameSupplier renameLabels_DefaultNameSupplier0 = new RenameLabels.DefaultNameSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(renameLabels_DefaultNameSupplier0, "com.google.javascript.jscomp.MakeDeclaredNamesUnique", false);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("com.google.javascript.jscomp.MakeDeclaredNamesUnique");
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }
}
