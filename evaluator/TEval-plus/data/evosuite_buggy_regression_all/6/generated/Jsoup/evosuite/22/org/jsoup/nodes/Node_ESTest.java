/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:33:39 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Node_ESTest extends Node_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("abs:i", "abs:i");
      Document document0 = Document.createShell("abs:i");
      // Undeclared exception!
      try { 
        document0.before((Node) textNode0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TextNode textNode0 = new TextNode("abs:i", "abs:i");
      textNode0.setBaseUri("abs:i");
      assertEquals("abs:i", textNode0.baseUri());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Document.createShell("abs:[@4`$^evxji");
      // Undeclared exception!
      try { 
        document0.after((Node) document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TextNode textNode0 = new TextNode("Tfr", "Tfr");
      // Undeclared exception!
      try { 
        textNode0.after("Tfr");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Document.createShell("Tfr");
      Element element0 = document0.head();
      assertNotNull(element0);
      
      Node node0 = element0.unwrap();
      assertNull(node0);
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("head", "head", "org.jsoup.nodes.Node", "cularr");
      String string0 = documentType0.toString();
      assertEquals("<!DOCTYPE head PUBLIC \"head\" \"org.jsoup.nodes.Node\">", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TextNode textNode0 = new TextNode("Tfr", "Tfr");
      Node node0 = textNode0.clone();
      assertNotSame(node0, textNode0);
      assertNotNull(node0);
      assertEquals(0, node0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("?", "?");
      Node node0 = textNode0.removeAttr("?");
      assertSame(textNode0, node0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TextNode textNode0 = new TextNode("ibs:i", "ibs:i");
      // Undeclared exception!
      try { 
        textNode0.siblingNodes();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Document.createShell("?");
      String string0 = document0.toString();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("abs:i", "abs:i");
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertEquals("abs:i", textNode0.baseUri());
      assertNotNull(textNode0);
      
      Attributes attributes0 = textNode0.attributes();
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertEquals("abs:i", textNode0.baseUri());
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Document.createShell("abs:class");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("abs:class", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertNotNull(document0);
      
      String string0 = document0.absUrl("abs:class");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("abs:class", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertNotNull(string0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = Document.createShell("abs:class");
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("abs:class", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      Document document1 = (Document)document0.addClass("abs:class");
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("abs:class", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document1.siblingIndex());
      assertEquals("#document", document1.nodeName());
      assertFalse(document1.isBlock());
      assertEquals("abs:class", document1.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertNotNull(document1);
      
      String string0 = document1.absUrl("abs:class");
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("abs:class", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document1.siblingIndex());
      assertEquals("#document", document1.nodeName());
      assertFalse(document1.isBlock());
      assertEquals("abs:class", document1.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertNotNull(string0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = Document.createShell("Tfr");
      assertFalse(document0.isBlock());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      document0.setParentNode(document0);
      assertFalse(document0.isBlock());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      
      Document document1 = (Document)document0.before("Tfr");
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertFalse(document0.isBlock());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document1.nodeName());
      assertFalse(document1.isBlock());
      assertEquals(0, document1.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("Tfr", document1.baseUri());
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = Document.createShell("$ln");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("$ln", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      Element element0 = document0.head();
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("$ln", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals("$ln", element0.baseUri());
      assertEquals(0, element0.siblingIndex());
      assertTrue(element0.isBlock());
      assertNotNull(element0);
      
      element0.replaceWith(document0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("$ln", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals("$ln", element0.baseUri());
      assertEquals(0, element0.siblingIndex());
      assertTrue(element0.isBlock());
      
      // Undeclared exception!
      try { 
        element0.wrap("$ln");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = Document.createShell("Tfr");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("Tfr", document0.baseUri());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      document0.setParentNode(document0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("Tfr", document0.baseUri());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      
      Document document1 = (Document)document0.wrap("<l.oB>*_S4E");
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("Tfr", document0.baseUri());
      assertEquals("#document", document0.nodeName());
      assertEquals(1, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertEquals("#document", document1.nodeName());
      assertEquals(1, document1.siblingIndex());
      assertFalse(document1.isBlock());
      assertEquals("Tfr", document1.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = Document.createShell("ibs:i");
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("ibs:i", document0.baseUri());
      assertNotNull(document0);
      
      Element element0 = document0.prependElement("ibs:i");
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("ibs:i", document0.baseUri());
      assertFalse(element0.isBlock());
      assertEquals("ibs:i", element0.baseUri());
      assertEquals(0, element0.siblingIndex());
      assertNotNull(element0);
      
      Node node0 = element0.wrap("ibs:i");
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("ibs:i", document0.baseUri());
      assertFalse(element0.isBlock());
      assertEquals("ibs:i", element0.baseUri());
      assertEquals(0, element0.siblingIndex());
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = Document.createShell("absXi");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("absXi", document0.baseUri());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      Element element0 = document0.head();
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("absXi", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("absXi", element0.baseUri());
      assertTrue(element0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertNotNull(element0);
      
      Element element1 = (Element)element0.wrap("absXi");
      assertSame(element0, element1);
      assertSame(element1, element0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("absXi", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("absXi", element0.baseUri());
      assertTrue(element0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertEquals("absXi", element1.baseUri());
      assertTrue(element1.isBlock());
      assertEquals(0, element1.siblingIndex());
      assertNotNull(element1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = Document.createShell("Tfr");
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      Document document1 = document0.normalise();
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document1.siblingIndex());
      assertEquals("#document", document1.nodeName());
      assertEquals("Tfr", document1.baseUri());
      assertFalse(document1.isBlock());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertNotNull(document1);
      
      document1.parentNode = (Node) document0;
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document1.siblingIndex());
      assertEquals("#document", document1.nodeName());
      assertEquals("Tfr", document1.baseUri());
      assertFalse(document1.isBlock());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals(0, document1.parentNode.siblingIndex());
      assertEquals("Tfr", document1.parentNode.baseUri());
      
      Element element0 = (Element)document0.unwrap();
      assertSame(document0, document1);
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, element0.siblingIndex());
      assertEquals("Tfr", element0.baseUri());
      assertTrue(element0.isBlock());
      assertNotNull(element0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = Document.createShell("$ln");
      assertEquals("$ln", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      Element element0 = document0.head();
      assertEquals("$ln", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, element0.siblingIndex());
      assertTrue(element0.isBlock());
      assertEquals("$ln", element0.baseUri());
      assertNotNull(element0);
      
      element0.setParentNode(document0);
      assertEquals("$ln", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, element0.siblingIndex());
      assertTrue(element0.isBlock());
      assertEquals("$ln", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TextNode textNode0 = new TextNode("Hdky;<-wU%c>", "Hdky;<-wU%c>");
      assertEquals("Hdky;<-wU%c>", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertNotNull(textNode0);
      
      // Undeclared exception!
      try { 
        textNode0.replaceChild(textNode0, textNode0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("", document0.baseUri());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      // Undeclared exception!
      try { 
        document0.removeChild(document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = Document.createShell("Tfr");
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      Element element0 = document0.head();
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertEquals("Tfr", element0.baseUri());
      assertTrue(element0.isBlock());
      assertNotNull(element0);
      
      Node node0 = element0.previousSibling();
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertEquals("Tfr", element0.baseUri());
      assertTrue(element0.isBlock());
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = Document.createShell("Tfr");
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("Tfr", document0.baseUri());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      Document document1 = document0.normalise();
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("Tfr", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("Tfr", document1.baseUri());
      assertEquals("#document", document1.nodeName());
      assertFalse(document1.isBlock());
      assertEquals(0, document1.siblingIndex());
      assertNotNull(document1);
      
      document1.parentNode = (Node) document0;
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("Tfr", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("Tfr", document1.baseUri());
      assertEquals("#document", document1.nodeName());
      assertFalse(document1.isBlock());
      assertEquals(0, document1.siblingIndex());
      assertEquals(0, document1.parentNode.siblingIndex());
      assertEquals("Tfr", document1.parentNode.baseUri());
      
      document0.setSiblingIndex(1230);
      assertSame(document0, document1);
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("Tfr", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals(1230, document0.siblingIndex());
      
      // Undeclared exception!
      try { 
        document0.previousSibling();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1229, Size: 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TextNode textNode0 = new TextNode("Tfr", "org.jsoup.nodes.Node$OuterHtmlVisitor");
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("org.jsoup.nodes.Node$OuterHtmlVisitor", textNode0.baseUri());
      assertNotNull(textNode0);
      
      boolean boolean0 = textNode0.equals(textNode0);
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("org.jsoup.nodes.Node$OuterHtmlVisitor", textNode0.baseUri());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TextNode textNode0 = new TextNode("Tfr", "Tfr");
      assertEquals("#text", textNode0.nodeName());
      assertEquals("Tfr", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertNotNull(textNode0);
      
      textNode0.hashCode();
      assertEquals("#text", textNode0.nodeName());
      assertEquals("Tfr", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = Document.createShell("Tfr");
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      Document document1 = document0.normalise();
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("Tfr", document1.baseUri());
      assertFalse(document1.isBlock());
      assertEquals(0, document1.siblingIndex());
      assertEquals("#document", document1.nodeName());
      assertNotNull(document1);
      
      document1.parentNode = (Node) document0;
      assertEquals("#document", document0.nodeName());
      assertEquals("Tfr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("Tfr", document1.baseUri());
      assertFalse(document1.isBlock());
      assertEquals(0, document1.siblingIndex());
      assertEquals("#document", document1.nodeName());
      assertEquals("Tfr", document1.parentNode.baseUri());
      assertEquals(0, document1.parentNode.siblingIndex());
      
      // Undeclared exception!
      try { 
        document1.parentNode.hashCode();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.nodes.Node");
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("org.jsoup.nodes.Node", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      document0.hashCode();
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("org.jsoup.nodes.Node", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = Document.createShell("abs:[@4`$^evxji");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertEquals("abs:[@4`$^evxji", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertNotNull(document0);
      
      Document document1 = (Document)document0.clone();
      assertNotSame(document0, document1);
      assertNotSame(document1, document0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertEquals("abs:[@4`$^evxji", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("#document", document1.nodeName());
      assertEquals(0, document1.siblingIndex());
      assertEquals("abs:[@4`$^evxji", document1.baseUri());
      assertFalse(document1.isBlock());
      assertNotNull(document1);
      assertFalse(document1.equals((Object)document0));
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TextNode textNode0 = new TextNode("Tfr", "Tfr");
      assertEquals("Tfr", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertNotNull(textNode0);
      
      String string0 = textNode0.toString();
      assertEquals("Tfr", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertNotNull(string0);
      assertEquals("Tfr", string0);
  }
}