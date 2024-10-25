/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:53:42 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.charset.Charset;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.helper.W3CDom;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Node;
import org.junit.runner.RunWith;
import org.w3c.dom.DOMException;
import org.w3c.dom.Document;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class W3CDom_ESTest extends W3CDom_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      String string0 = w3CDom0.asString((Document) null);
      assertEquals("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("");
      w3CDom0.fromJsoup(document0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("p&{mv:XEA/,^95|:");
      document0.text("p&{mv:XEA/,^95|:");
      Document document1 = w3CDom0.fromJsoup(document0);
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("i(l~`V5#K=I2#");
      Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      DataNode dataNode0 = DataNode.createFromEncoded("i(l~`V5#K=I2#", "i(l~`V5#K=I2#");
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(dataNode0, 1921);
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
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("i.l~`V5#K=I2#");
      Document document1 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document1);
      Comment comment0 = new Comment("i.l~`V5#K=I2#");
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(comment0, 1400);
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
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder((Document) null);
      w3CDom_W3CBuilder0.head((Node) null, (-1));
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = org.jsoup.nodes.Document.createShell("i.l~`V5#K=I2#");
      Charset charset0 = document0.charset();
      document0.charset(charset0);
      Document document1 = w3CDom0.fromJsoup(document0);
      assertNotNull(document1);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      org.jsoup.nodes.Document document0 = new org.jsoup.nodes.Document("i.A~`V]5#K=I2#");
      org.jsoup.nodes.Document document1 = document0.normalise();
      W3CDom w3CDom0 = new W3CDom();
      document1.attr("xmlns", "xmlns");
      Document document2 = w3CDom0.fromJsoup(document0);
      W3CDom.W3CBuilder w3CDom_W3CBuilder0 = new W3CDom.W3CBuilder(document2);
      // Undeclared exception!
      try { 
        w3CDom_W3CBuilder0.head(document0, (-158656234));
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      W3CDom w3CDom0 = new W3CDom();
      org.jsoup.nodes.Document document0 = new org.jsoup.nodes.Document("[a-zA-Z_:][-a-zA-Z0-9_:.]*");
      document0.appendElement("[a-zA-Z_:][-a-zA-Z0-9_:.]*");
      // Undeclared exception!
      try { 
        w3CDom0.fromJsoup(document0);
        fail("Expecting exception: DOMException");
      
      } catch(DOMException e) {
      }
  }
}
