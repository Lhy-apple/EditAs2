/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:34:34 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.sun.org.apache.xerces.internal.dom.DocumentImpl;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.helper.W3CDom;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Node;
import org.junit.runner.RunWith;
import org.w3c.dom.DOMException;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class W3CDom_ESTest extends W3CDom_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = new Document("");
      Document document1 = document0.normalise();
      org.w3c.dom.Document document2 = w3CDom0.fromJsoup(document1);
      String string0 = w3CDom0.asString(document2);
      assertEquals("<html>\n<head>\n<META http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">\n</head>\n<body></body>\n</html>\n", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Document.createShell("<!Udoctype");
      Document document1 = (Document)document0.text(":matches(%s");
      DocumentImpl documentImpl0 = (DocumentImpl)w3CDom0.fromJsoup(document1);
      assertNotNull(documentImpl0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Document.createShell("3u#SWpzZWBeBK0k");
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      w3CDom_W3CBuilder0.head((Node) null, 1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder((org.w3c.dom.Document) null);
      Comment comment0 = new Comment(":", ":");
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(comment0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.W3CDom$W3CBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Document.createShell("<!doctype");
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      DataNode dataNode0 = DataNode.createFromEncoded("<!doctype", "<!doctype");
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(dataNode0, 260);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.W3CDom$W3CBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Document.createShell("<!Udoctype");
      Charset charset0 = document0.charset();
      document0.charset(charset0);
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Document.createShell("xmlns");
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      document0.attr("xmlns", true);
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(document0, 1);
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder((org.w3c.dom.Document) null);
      Document document0 = Document.createShell("Gb>M69");
      document0.attr("xmlns:wf3cekw#sgty", true);
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(document0, (-3826));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.helper.W3CDom$W3CBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      Document document0 = Document.createShell("[^-a-zA-Z0-9_:.]");
      org.w3c.dom.Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      document0.tagName("[^-a-zA-Z0-9_:.]");
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(document0, (-295));
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }
}
