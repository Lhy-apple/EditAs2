/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:27:01 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.CDataNode;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.TextNode;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LeafNode_ESTest extends LeafNode_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode("SKIP_CHILDREN");
      TextNode textNode0 = cDataNode0.text("d3=2bDkSIs:[j");
      assertSame(cDataNode0, textNode0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Comment comment0 = new Comment("", "/C;");
      String string0 = comment0.toString();
      assertEquals("\n<!---->", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      TextNode textNode0 = new TextNode("org.jsoup.nodes.LeafNode", "!9$s");
      // Undeclared exception!
      try { 
        textNode0.removeAttr((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode((String) null);
      // Undeclared exception!
      try { 
        cDataNode0.childNodesAsArray();
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
      DataNode dataNode0 = new DataNode("org.jsoup.nodes.LeafNode", "org.jsoup.nodes.LeafNode");
      String string0 = dataNode0.absUrl("j_`m9h;QV");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode(">=5~ma}VJv`Z7+mI");
      cDataNode0.setBaseUri("k");
      assertEquals(0, cDataNode0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("#text", "Leaf Nodes do not have child nodes.", "Leaf Nodes do not have child nodes.");
      assertFalse(documentType0.hasParent());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode("SKIP_CHILDREN");
      String string0 = cDataNode0.attr("Leaf Nodes do not have child nodes.");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode(">=5~ma}VJv`Z7+mI");
      String string0 = cDataNode0.baseUri();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode(">=5~ma}VJv`Z7+mI");
      cDataNode0.reparentChild(cDataNode0);
      // Undeclared exception!
      try { 
        cDataNode0.baseUri();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}