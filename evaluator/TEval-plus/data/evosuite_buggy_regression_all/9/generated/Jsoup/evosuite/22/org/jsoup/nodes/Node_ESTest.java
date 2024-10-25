/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:05:42 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
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
      Document document0 = Document.createShell("6");
      // Undeclared exception!
      try { 
        document0.before((Node) document0);
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
      DataNode dataNode0 = new DataNode("gnap", "gnap");
      dataNode0.setBaseUri("gnap");
      assertEquals(0, dataNode0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Document.createShell("6");
      DataNode dataNode0 = DataNode.createFromEncoded("6", "6");
      // Undeclared exception!
      try { 
        document0.after((Node) dataNode0);
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
      TextNode textNode0 = TextNode.createFromEncoded("Mcy", "Mcy");
      Node node0 = textNode0.clone();
      assertNotNull(node0);
      assertNotSame(node0, textNode0);
      assertEquals(0, node0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder(";[v:,UA%scdt");
      DocumentType documentType0 = new DocumentType(";[v:,UA%scdt", "em,QcAx/(J", "&j:u+*\"", "&j:u+*\"");
      stringBuilder0.append((Object) documentType0);
      assertEquals(";[v:,UA%scdt<!DOCTYPE ;[v:,UA%scdt PUBLIC \"em,QcAx/(J\" \"&j:u+*\"\">", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Document.createShell("='!HB2M!|xO,[;I");
      document0.prependChild(document0);
      Node node0 = document0.unwrap();
      assertEquals(0, document0.siblingIndex());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("Mcy", "Mcy");
      // Undeclared exception!
      try { 
        textNode0.before("Mcy");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TextNode textNode0 = new TextNode("abs:output", "abs:output");
      Node node0 = textNode0.removeAttr("<*iy_)z&Q.uH~;\"/");
      assertEquals("#text", node0.nodeName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("Mcy", "Mcy");
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
      Document document0 = Document.createShell("");
      String string0 = document0.outerHtml();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("Mcy", "Mcy");
      assertEquals("Mcy", textNode0.baseUri());
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertNotNull(textNode0);
      
      Attributes attributes0 = textNode0.attributes();
      assertEquals("Mcy", textNode0.baseUri());
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("abs:#text", "abs:#text");
      assertEquals("abs:#text", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertNotNull(textNode0);
      
      String string0 = textNode0.absUrl("abs:abs:#text");
      assertEquals("abs:#text", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertNotNull(string0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("abs:#text", "abs:#text");
      assertEquals("abs:#text", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertNotNull(textNode0);
      
      Node node0 = textNode0.attr("abs:#text", "abs:#text");
      assertSame(textNode0, node0);
      assertSame(node0, textNode0);
      assertEquals("abs:#text", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, node0.siblingIndex());
      assertEquals("#text", node0.nodeName());
      assertEquals("abs:#text", node0.baseUri());
      assertNotNull(node0);
      
      String string0 = node0.absUrl("abs:abs:#text");
      assertSame(textNode0, node0);
      assertSame(node0, textNode0);
      assertEquals("abs:#text", textNode0.baseUri());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, node0.siblingIndex());
      assertEquals("#text", node0.nodeName());
      assertEquals("abs:#text", node0.baseUri());
      assertNotNull(string0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TextNode textNode0 = new TextNode("ab:eG91bdlr", "ab:eG91bdlr");
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertEquals("ab:eG91bdlr", textNode0.baseUri());
      assertNotNull(textNode0);
      
      DataNode dataNode0 = DataNode.createFromEncoded("ab:eG91bdlr", "j2");
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("j2", dataNode0.baseUri());
      assertNotNull(dataNode0);
      
      dataNode0.parentNode = (Node) textNode0;
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("#text", textNode0.nodeName());
      assertEquals("ab:eG91bdlr", textNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("j2", dataNode0.baseUri());
      assertEquals("ab:eG91bdlr", dataNode0.parentNode.baseUri());
      assertEquals(0, dataNode0.parentNode.siblingIndex());
      
      // Undeclared exception!
      try { 
        dataNode0.after("ab:eG91bdlr");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DataNode dataNode0 = DataNode.createFromEncoded("LJ", "LJ");
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("LJ", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertNotNull(dataNode0);
      
      Document document0 = Document.createShell("LJ");
      assertFalse(document0.isBlock());
      assertEquals("LJ", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertNotNull(document0);
      
      dataNode0.parentNode = (Node) document0;
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("LJ", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals("LJ", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, dataNode0.parentNode.siblingIndex());
      assertEquals("LJ", dataNode0.parentNode.baseUri());
      
      Node node0 = dataNode0.after("LJ");
      assertSame(dataNode0, node0);
      assertSame(node0, dataNode0);
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("LJ", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(0, node0.siblingIndex());
      assertEquals("#data", node0.nodeName());
      assertEquals("LJ", node0.baseUri());
      assertNotNull(node0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = Document.createShell("!bs:-gext");
      assertEquals("!bs:-gext", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertNotNull(document0);
      
      Element element0 = document0.head();
      assertEquals("!bs:-gext", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals(0, element0.siblingIndex());
      assertTrue(element0.isBlock());
      assertEquals("!bs:-gext", element0.baseUri());
      assertNotNull(element0);
      
      element0.replaceWith(document0);
      assertEquals("!bs:-gext", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals(0, element0.siblingIndex());
      assertTrue(element0.isBlock());
      assertEquals("!bs:-gext", element0.baseUri());
      
      // Undeclared exception!
      try { 
        element0.wrap("!bs:-gext");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = Document.createShell("abs:st.g]z&v(m~wm");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("abs:st.g]z&v(m~wm", document0.baseUri());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      Element element0 = document0.head();
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("abs:st.g]z&v(m~wm", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("abs:st.g]z&v(m~wm", element0.baseUri());
      assertTrue(element0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertNotNull(element0);
      
      Element element1 = (Element)element0.wrap("abs:st.g]z&v(m~wm");
      assertSame(element0, element1);
      assertSame(element1, element0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals("abs:st.g]z&v(m~wm", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("abs:st.g]z&v(m~wm", element0.baseUri());
      assertTrue(element0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertEquals(0, element1.siblingIndex());
      assertEquals("abs:st.g]z&v(m~wm", element1.baseUri());
      assertTrue(element1.isBlock());
      assertNotNull(element1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = Document.createShell("ab:eG91lb)lr");
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals("ab:eG91lb)lr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      Element element0 = document0.prependElement("ab:eG91lb)lr");
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals("ab:eG91lb)lr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("ab:eG91lb)lr", element0.baseUri());
      assertFalse(element0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertNotNull(element0);
      
      Node node0 = element0.wrap("ab:eG91lb)lr");
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals("ab:eG91lb)lr", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("ab:eG91lb)lr", element0.baseUri());
      assertFalse(element0.isBlock());
      assertEquals(0, element0.siblingIndex());
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("='!H:2M!0xO,-e[iI");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("='!H:2M!0xO,-e[iI", document0.baseUri());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertNotNull(document0);
      
      Document document1 = (Document)document0.prependChild(document0);
      assertSame(document0, document1);
      assertSame(document1, document0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("='!H:2M!0xO,-e[iI", document0.baseUri());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document1.nodeName());
      assertEquals(0, document1.siblingIndex());
      assertFalse(document1.isBlock());
      assertEquals("='!H:2M!0xO,-e[iI", document1.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertNotNull(document1);
      
      Document document2 = (Document)document0.empty();
      assertSame(document0, document2);
      assertSame(document0, document1);
      assertSame(document2, document0);
      assertSame(document2, document1);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("='!H:2M!0xO,-e[iI", document0.baseUri());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document2.nodeName());
      assertFalse(document2.isBlock());
      assertEquals(0, document2.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document2.quirksMode());
      assertEquals("='!H:2M!0xO,-e[iI", document2.baseUri());
      assertNotNull(document2);
      
      // Undeclared exception!
      try { 
        document0.unwrap();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = Document.createShell("abs:#tet");
      assertEquals("abs:#tet", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      Element element0 = document0.body();
      assertEquals("abs:#tet", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("abs:#tet", element0.baseUri());
      assertTrue(element0.isBlock());
      assertEquals(1, element0.siblingIndex());
      assertNotNull(element0);
      
      element0.setParentNode(document0);
      assertEquals("abs:#tet", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("abs:#tet", element0.baseUri());
      assertTrue(element0.isBlock());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.nodes.Node");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertFalse(document0.isBlock());
      assertEquals("org.jsoup.nodes.Node", document0.baseUri());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertNotNull(document0);
      
      // Undeclared exception!
      try { 
        document0.replaceChild(document0, document0);
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
      Document document0 = Document.createShell("!bs:-gext");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertFalse(document0.isBlock());
      assertEquals(0, document0.siblingIndex());
      assertEquals("!bs:-gext", document0.baseUri());
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
      DataNode dataNode0 = DataNode.createFromEncoded("LJ", "LJ");
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("LJ", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertNotNull(dataNode0);
      
      Document document0 = Document.createShell("LJ");
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertEquals("LJ", document0.baseUri());
      assertNotNull(document0);
      
      dataNode0.parentNode = (Node) document0;
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("LJ", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertFalse(document0.isBlock());
      assertEquals("LJ", document0.baseUri());
      assertEquals("LJ", dataNode0.parentNode.baseUri());
      assertEquals(0, dataNode0.parentNode.siblingIndex());
      
      Node node0 = dataNode0.previousSibling();
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("LJ", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      DataNode dataNode0 = DataNode.createFromEncoded("='!HB2M!|xO,[;I", "='!HB2M!|xO,[;I");
      assertEquals("#data", dataNode0.nodeName());
      assertEquals("='!HB2M!|xO,[;I", dataNode0.baseUri());
      assertEquals(0, dataNode0.siblingIndex());
      assertNotNull(dataNode0);
      
      Document document0 = Document.createShell("dwGyhFO525QhyLSC");
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("dwGyhFO525QhyLSC", document0.baseUri());
      assertFalse(document0.isBlock());
      assertNotNull(document0);
      
      dataNode0.setSiblingIndex(1196);
      assertEquals("#data", dataNode0.nodeName());
      assertEquals("='!HB2M!|xO,[;I", dataNode0.baseUri());
      assertEquals(1196, dataNode0.siblingIndex());
      
      dataNode0.parentNode = (Node) document0;
      assertEquals("#data", dataNode0.nodeName());
      assertEquals("='!HB2M!|xO,[;I", dataNode0.baseUri());
      assertEquals(1196, dataNode0.siblingIndex());
      assertEquals(0, document0.siblingIndex());
      assertEquals("#document", document0.nodeName());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals("dwGyhFO525QhyLSC", document0.baseUri());
      assertFalse(document0.isBlock());
      assertEquals("dwGyhFO525QhyLSC", dataNode0.parentNode.baseUri());
      assertEquals(0, dataNode0.parentNode.siblingIndex());
      
      // Undeclared exception!
      try { 
        dataNode0.previousSibling();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1195, Size: 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      DataNode dataNode0 = DataNode.createFromEncoded("Mcy", "Mcy");
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("Mcy", dataNode0.baseUri());
      assertNotNull(dataNode0);
      
      boolean boolean0 = dataNode0.equals("Mcy");
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(0, dataNode0.siblingIndex());
      assertEquals("Mcy", dataNode0.baseUri());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      DataNode dataNode0 = DataNode.createFromEncoded("='!K2M!|xgO,[II", "='!K2M!|xgO,[II");
      assertEquals("='!K2M!|xgO,[II", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(0, dataNode0.siblingIndex());
      assertNotNull(dataNode0);
      
      boolean boolean0 = dataNode0.equals(dataNode0);
      assertEquals("='!K2M!|xgO,[II", dataNode0.baseUri());
      assertEquals("#data", dataNode0.nodeName());
      assertEquals(0, dataNode0.siblingIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("My", "My");
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("My", textNode0.baseUri());
      assertNotNull(textNode0);
      
      textNode0.hashCode();
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertEquals("My", textNode0.baseUri());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = Document.createShell("");
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertNotNull(document0);
      
      Comment comment0 = new Comment("abs:org.jsoup.nodes.node", "");
      assertEquals(0, comment0.siblingIndex());
      assertEquals("#comment", comment0.nodeName());
      assertEquals("", comment0.baseUri());
      assertNotNull(comment0);
      
      Document document1 = (Document)document0.doClone(comment0);
      assertNotSame(document0, document1);
      assertNotSame(document1, document0);
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, comment0.siblingIndex());
      assertEquals("#comment", comment0.nodeName());
      assertEquals("", comment0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("", document1.baseUri());
      assertFalse(document1.isBlock());
      assertEquals("#document", document1.nodeName());
      assertEquals(0, document1.siblingIndex());
      assertNotNull(document1);
      assertFalse(document1.equals((Object)document0));
      
      document1.hashCode();
      assertNotSame(document0, document1);
      assertNotSame(document1, document0);
      assertFalse(document0.isBlock());
      assertEquals("#document", document0.nodeName());
      assertEquals(0, document0.siblingIndex());
      assertEquals("", document0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
      assertEquals(0, comment0.siblingIndex());
      assertEquals("#comment", comment0.nodeName());
      assertEquals("", comment0.baseUri());
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
      assertEquals("", document1.baseUri());
      assertFalse(document1.isBlock());
      assertEquals("#document", document1.nodeName());
      assertEquals(0, document1.siblingIndex());
      assertFalse(document0.equals((Object)document1));
      assertFalse(document1.equals((Object)document0));
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("Mcy", "Mcy");
      assertEquals("Mcy", textNode0.baseUri());
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertNotNull(textNode0);
      
      String string0 = textNode0.toString();
      assertEquals("Mcy", textNode0.baseUri());
      assertEquals("#text", textNode0.nodeName());
      assertEquals(0, textNode0.siblingIndex());
      assertNotNull(string0);
      assertEquals("Mcy", string0);
  }
}
