/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:57:58 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.CDataNode;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LeafNode_ESTest extends LeafNode_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      TextNode textNode0 = new TextNode(";Lr|rOP\"T");
      String string0 = textNode0.absUrl("c(<");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Comment comment0 = new Comment("");
      String string0 = comment0.toString();
      assertEquals("\n<!---->", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode("s^_Dde3I$8U");
      Node node0 = cDataNode0.removeAttr("<![CDATA[s^_Dde3I$8U]]>");
      assertEquals(0, node0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      TextNode textNode0 = new TextNode(";Lr|rOP\"T");
      // Undeclared exception!
      try { 
        textNode0.ensureChildNodes();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Leaf Nodes do not have child nodes.
         //
         verifyException("org.jsoup.nodes.LeafNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      TextNode textNode0 = TextNode.createFromEncoded("org.jsoup.nodes.LeafNode");
      TextNode textNode1 = textNode0.text("uMB3kIe7'?");
      assertEquals(0, textNode1.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Comment comment0 = new Comment("");
      comment0.doSetBaseUri("");
      assertEquals(0, comment0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("source", "IsaG`iBpReIjr", ";Lr|rOP\"T");
      assertFalse(documentType0.hasParent());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode("org.jsoup.nodes.Attributes");
      String string0 = cDataNode0.attr("org.jsoup.nodes.Attributes");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      TextNode textNode0 = new TextNode(";Lr|rOP\"T");
      String string0 = textNode0.baseUri();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      TextNode textNode0 = new TextNode(";Lr|rOP\"T");
      Document document0 = new Document("source");
      textNode0.parentNode = (Node) document0;
      String string0 = textNode0.baseUri();
      assertEquals("source", string0);
  }
}
