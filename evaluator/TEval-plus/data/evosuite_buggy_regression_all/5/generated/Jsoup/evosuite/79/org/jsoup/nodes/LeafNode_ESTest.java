/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:16:33 GMT 2023
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
      DocumentType documentType0 = new DocumentType("declare", "declare", "~9ARt", "z");
      documentType0.setBaseUri("#comment");
      assertEquals("#doctype", documentType0.nodeName());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      TextNode textNode0 = new TextNode("declare");
      TextNode textNode1 = textNode0.splitText(0);
      assertFalse(textNode1.equals((Object)textNode0));
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode("s^_Dde3I$8U");
      Node node0 = cDataNode0.removeAttr("<![CDATA[s^_Dde3I$8U]]>");
      assertEquals(0, node0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      TextNode textNode0 = new TextNode("declare");
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
      TextNode textNode0 = TextNode.createFromEncoded("org.jsoup.nodes.LeafNode", "org.jsoup.nodes.LeafNode");
      boolean boolean0 = textNode0.hasAttr("org.jsoup.nodes.LeafNode");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Comment comment0 = new Comment("");
      // Undeclared exception!
      try { 
        comment0.absUrl("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode("org.jsoup.nodes.Attributes");
      String string0 = cDataNode0.attr("org.jsoup.nodes.Attributes");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      TextNode textNode0 = new TextNode("declare");
      String string0 = textNode0.baseUri();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      TextNode textNode0 = new TextNode("declare");
      Document document0 = Document.createShell("declare");
      textNode0.parentNode = (Node) document0;
      String string0 = textNode0.baseUri();
      assertEquals("declare", string0);
  }
}