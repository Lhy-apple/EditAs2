/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:36:26 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.nodes.XmlDeclaration;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Node_ESTest extends Node_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("abs:)0x||6");
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
      TextNode textNode0 = TextNode.createFromEncoded("abs:)0||", "abs:)0||");
      textNode0.setBaseUri("abs:)0||");
      assertEquals("abs:)0||", textNode0.baseUri());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("", "");
      // Undeclared exception!
      try { 
        textNode0.after((Node) textNode0);
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
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("&|O", "&|O", false);
      String string0 = xmlDeclaration0.attr("abs:<(k");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("Q+B-i|", "Q+B-i|");
      textNode0.toString();
      assertEquals(0, textNode0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("abs:)x||6", "abs:)x||6");
      textNode0.setParentNode(textNode0);
      // Undeclared exception!
      try { 
        textNode0.unwrap();
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      DocumentType documentType0 = new DocumentType("+cLv~YYk4wOOK{'", "+cLv~YYk4wOOK{'", "2=\u0004@", "t}NNLP");
      // Undeclared exception!
      try { 
        stringBuilder0.insert((-1706), (Object) documentType0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
         //
         // String index out of range: -1706
         //
         verifyException("java.lang.AbstractStringBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("abs:)0x||6");
      // Undeclared exception!
      try { 
        document0.before("abs:)0x||6");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("p", "p");
      Node node0 = textNode0.removeAttr("p");
      assertEquals(0, node0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Document.createShell("abs:)0x||6");
      String string0 = document0.outerHtml();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("6", "6");
      String string0 = textNode0.attr("6");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Document.createShell("abs:)0x||6");
      String string0 = document0.absUrl("abs:)0x||6");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = Document.createShell("abs:)0x||6");
      document0.attr("abs:)0x||6", "abs:)0x||6");
      String string0 = document0.absUrl("abs:)0x||6");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = Document.createShell("abs:");
      List<Node> list0 = document0.childNodesCopy();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = Document.createShell("^e/g");
      Element element0 = document0.head();
      Element element1 = element0.wrap("^e/g");
      assertEquals(0, element0.siblingIndex());
      assertNotNull(element1);
      assertEquals(0, element1.childNodeSize());
      
      document0.hashCode();
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TextNode textNode0 = new TextNode("ab.!)0x||6", "ab.!)0x||6");
      Node node0 = textNode0.wrap("<!doctype");
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("6", "6");
      Node[] nodeArray0 = new Node[1];
      nodeArray0[0] = (Node) textNode0;
      textNode0.addChildren(nodeArray0);
      textNode0.wrap("6");
      assertEquals(1, textNode0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = Document.createShell("abs:");
      document0.prependChild(document0);
      document0.unwrap();
      assertEquals(1, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("abs)0x|6");
      Document document1 = (Document)document0.prependChild(document0);
      document1.setParentNode(document1);
      assertSame(document0, document1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = Document.createShell("abs:");
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
  public void test20()  throws Throwable  {
      TextNode textNode0 = new TextNode("ab.!)0x|6", "ab.!)0x|6");
      Document document0 = new Document("ab.!)0x|6");
      Element element0 = document0.appendChild(textNode0);
      textNode0.replaceWith(element0);
      assertEquals(0, element0.siblingIndex());
      assertEquals(0, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("abs:tU5X", "abs:tU5X");
      Document document0 = new Document("abs:tU5X");
      // Undeclared exception!
      try { 
        textNode0.removeChild(document0);
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
      Document document0 = Document.createShell("vbQ|^wc9F91");
      List<Node> list0 = document0.siblingNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("abs)0x|6");
      document0.prependChild(document0);
      Element element0 = document0.after("abs)0x|6");
      List<Node> list0 = element0.siblingNodes();
      assertEquals(1, list0.size());
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = Document.createShell("b.!)D75x||6");
      Element element0 = document0.prependChild(document0);
      Node node0 = element0.previousSibling();
      assertNull(node0);
      assertEquals(0, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = Document.createShell(">=Id>2c{wIKf_r=3[/");
      Node node0 = document0.previousSibling();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = Document.createShell("b.!)D75x||6");
      Node[] nodeArray0 = new Node[4];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      nodeArray0[2] = (Node) document0;
      nodeArray0[3] = (Node) document0;
      document0.addChildren(nodeArray0);
      document0.previousSibling();
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("&|O", "&|O", false);
      boolean boolean0 = xmlDeclaration0.equals("&|O");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TextNode textNode0 = new TextNode("6", "6");
      TextNode textNode1 = TextNode.createFromEncoded("h`bOk!0U9OB$kPECguy", "abs:f1G");
      textNode0.childNodes = null;
      boolean boolean0 = textNode0.equals(textNode1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("6", "6");
      TextNode textNode1 = new TextNode("6", "6");
      assertTrue(textNode1.equals((Object)textNode0));
      
      textNode1.childNodes = null;
      boolean boolean0 = textNode0.equals(textNode1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TextNode textNode0 = new TextNode("6", "6");
      textNode0.childNodes = null;
      TextNode textNode1 = new TextNode("6", "6");
      assertFalse(textNode1.equals((Object)textNode0));
      
      textNode1.childNodes = null;
      boolean boolean0 = textNode0.equals(textNode1);
      assertTrue(textNode1.equals((Object)textNode0));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TextNode textNode0 = new TextNode("6", "6");
      textNode0.hasAttr("org.jsoup.HttpStausException");
      textNode0.childNodes = null;
      TextNode textNode1 = new TextNode("6", "6");
      textNode1.childNodes = null;
      boolean boolean0 = textNode0.equals(textNode1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TextNode textNode0 = new TextNode(">", ">");
      TextNode textNode1 = TextNode.createFromEncoded(">", ">");
      Attributes attributes0 = textNode0.attributes();
      assertFalse(textNode0.equals((Object)textNode1));
      
      textNode1.attributes = attributes0;
      boolean boolean0 = textNode0.equals(textNode1);
      assertTrue(textNode1.equals((Object)textNode0));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("6", "6");
      TextNode textNode1 = new TextNode("6", "6");
      assertTrue(textNode1.equals((Object)textNode0));
      
      textNode1.attributes();
      boolean boolean0 = textNode0.equals(textNode1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = Document.createShell("^e/g");
      Element element0 = document0.head();
      element0.childNodes = null;
      document0.hashCode();
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("Q+B-i|", "Q+B-i|");
      Node node0 = textNode0.clone();
      assertNotSame(node0, textNode0);
      assertNotNull(node0);
      assertEquals(0, node0.siblingIndex());
  }
}