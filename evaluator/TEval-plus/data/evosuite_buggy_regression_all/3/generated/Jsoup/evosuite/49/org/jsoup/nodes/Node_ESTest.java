/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:53:04 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
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
      Document document0 = Document.createShell("abs:yw>pl2");
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
      Document document0 = new Document("abs:yw>pl2");
      document0.setBaseUri("abs:yw>pl2");
      assertFalse(document0.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      // Undeclared exception!
      try { 
        document0.after((Node) null);
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
      Comment comment0 = new Comment("declaration", "declaration");
      // Undeclared exception!
      try { 
        comment0.removeChild(comment0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("CM6/0P<Z");
      document0.prependChild(document0);
      Node node0 = document0.unwrap();
      assertEquals("CM6/0P<Z", node0.baseUri());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("org.jsoup.nodes.Node", "org.jsoup.nodes.Node", "body", "dg0Wnr b#%hjhwdB");
      String string0 = documentType0.toString();
      assertEquals("<!DOCTYPE org.jsoup.nodes.Node PUBLIC \"org.jsoup.nodes.Node\" \"body\">", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = Document.createShell("");
      // Undeclared exception!
      try { 
        document0.removeAttr("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Comment comment0 = new Comment("A3!uNcNIfzX[J.tV}H", "A3!uNcNIfzX[J.tV}H");
      Node[] nodeArray0 = new Node[1];
      nodeArray0[0] = (Node) comment0;
      comment0.addChildren(nodeArray0);
      comment0.before("A3!uNcNIfzX[J.tV}H");
      assertEquals(2, comment0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      String string0 = document0.outerHtml();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TextNode textNode0 = new TextNode("abs:ywvpl2", "abs:ywvpl2");
      textNode0.outerHtml();
      assertEquals(0, textNode0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      Attributes attributes0 = document0.attributes();
      assertNotNull(attributes0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      String string0 = document0.attr("abs:yw>pl2");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("dbs:yw>p");
      String string0 = document0.attr("dbs:yw>p");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      String string0 = document0.absUrl("abs:yw>pl2");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      document0.attr("abs:yw>pl2", "abs:yw>pl2");
      String string0 = document0.absUrl("abs:yw>pl2");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      List<Node> list0 = document0.childNodesCopy();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      Node node0 = document0.doClone(document0);
      assertTrue(node0.equals((Object)document0));
      assertNotSame(node0, document0);
      
      node0.replaceWith(document0);
      assertFalse(node0.equals((Object)document0));
      
      document0.wrap("abs:yw>pl2");
      assertFalse(document0.equals((Object)node0));
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Comment comment0 = new Comment("A3!uNcNIfzX[J.tV}H", "A3!uNcNIfzX[J.tV}H");
      Node[] nodeArray0 = new Node[1];
      nodeArray0[0] = (Node) comment0;
      comment0.addChildren(nodeArray0);
      assertEquals(1, comment0.childNodeSize());
      
      comment0.wrap("A3!uNcNIfzX[J.tV}H");
      assertEquals(0, comment0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("");
      Document.QuirksMode document_QuirksMode0 = Document.QuirksMode.noQuirks;
      Document document1 = document0.quirksMode(document_QuirksMode0);
      document1.parentNode = (Node) document0;
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
      Document document0 = Document.createShell("ir7IDI");
      Element element0 = document0.prependChild(document0);
      document0.setParentNode(element0);
      assertEquals(0, element0.siblingIndex());
      assertEquals(0, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("abs:yw>pl2");
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
      Document document0 = Document.createShell("Rl");
      document0.prependChild(document0);
      List<Node> list0 = document0.siblingNodes();
      assertEquals(0, document0.siblingIndex());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = Document.createShell("abs:yw>pl2");
      List<Node> list0 = document0.siblingNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("KUd6F");
      document0.prependChild(document0);
      Node node0 = document0.previousSibling();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("body");
      Node node0 = document0.previousSibling();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("inva");
      Document document1 = document0.normalise();
      document0.appendChild(document1);
      Node node0 = document0.previousSibling();
      assertEquals(2, node0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Comment comment0 = new Comment("declaration", "declaration");
      Comment comment1 = new Comment("declaration", "declaration");
      comment0.equals(comment1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Comment comment0 = new Comment("declaration", "declaration");
      comment0.equals(comment0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Comment comment0 = new Comment("declaration", "declaration");
      comment0.equals((Object) null);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Comment comment0 = new Comment("absNywHpl2", "absNywHpl2");
      comment0.equals("absNywHpl2");
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Comment comment0 = new Comment("declaration", "declaration");
      Document document0 = Document.createShell("declaration");
      List<Node> list0 = document0.childNodes();
      comment0.childNodes = list0;
      Comment comment1 = new Comment("declaration", "declaration");
      comment1.equals(comment0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Comment comment0 = new Comment("declaration", "declaration");
      comment0.attributes = null;
      Comment comment1 = new Comment("declaration", "declaration");
      comment0.equals(comment1);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Comment comment0 = new Comment("declaration", "declaration");
      Comment comment1 = new Comment("abs:%,'5SgQ:", "abs:%,'5SgQ:");
      comment0.equals(comment1);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Comment comment0 = new Comment("declaration", "declaration");
      comment0.attributes = null;
      Comment comment1 = new Comment("declaration", "declaration");
      comment1.attributes = null;
      comment0.equals(comment1);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("6bs:D/Hlh");
      document0.childNodes = null;
      document0.hashCode();
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = Document.createShell("declaration");
      Element element0 = document0.append("declaration");
      element0.hashCode();
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("abs:yw>pl2");
      Document document1 = (Document)document0.prependChild(document0);
      document0.after("abs:yw>pl2");
      document1.clone();
  }
}
